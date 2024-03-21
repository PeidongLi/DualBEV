"""
Copyright (c) Zhijia Technology. All rights reserved.

Author: Peidong Li (lipeidong@smartxtruck.com / peidongl@outlook.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmcv.runner import BaseModule, force_fp32
from torch.cuda.amp.autocast_mode import autocast
from torch.utils.checkpoint import checkpoint
from mmdet.models.utils import LearnedPositionalEncoding
from .view_transformer import DepthNet, LSSViewTransformerBEVDepth
from mmdet3d.ops.bev_pool_v2.bev_pool import bev_pool_v2
from mmdet.models.backbones.resnet import BasicBlock
from ..builder import NECKS

class MS_CAM(nn.Module):
    "From https://github.com/YimianDai/open-aff/blob/master/aff_pytorch/aff_net/fusion.py"
    def __init__(self, input_channel=64, output_channel=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(input_channel // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(input_channel, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, output_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(output_channel),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channel, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, output_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(output_channel),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        return self.sigmoid(xlg)

class ChannelAttention(nn.Module):

    def __init__(self, input_channel, output_channel, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(input_channel, input_channel // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(input_channel // ratio, output_channel, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ResCBAMBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResCBAMBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes,planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ProbNet(BaseModule):

    def __init__(
        self,
        in_channels=512,
        scale_factor=1,
        with_centerness=False,
        loss_weight=6.0,
        bev_size=None,
    ):
        super(ProbNet, self).__init__()
        self.loss_weight=loss_weight
        mid_channels=in_channels//2
        self.base_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.prob_conv = nn.Sequential(
            ResCBAMBlock(mid_channels, mid_channels),      
            )
        self.mask_net = nn.Conv2d(mid_channels, 1, kernel_size=1, padding=0, stride=1)

        self.with_centerness=with_centerness
        if with_centerness:
            self.centerness = bev_centerness_weight(bev_size[0],bev_size[1]).cuda()
        self.dice_loss = DiceLoss(use_sigmoid=True, loss_weight=self.loss_weight)
        self.ce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.13]))

    def forward(self, input):
        height_feat = self.base_conv(input)
        height_feat = self.prob_conv(height_feat)            
        bev_prob = self.mask_net(height_feat)            
        return bev_prob

    def get_bev_mask_loss(self, gt_bev_mask, pred_bev_mask):
        bs, bev_h, bev_w = gt_bev_mask.shape
        b = gt_bev_mask.reshape(bs , bev_w * bev_h).permute(1, 0).to(torch.float)
        a = pred_bev_mask.reshape(bs, bev_w * bev_h).permute(1, 0)
        if self.with_centerness:
            self.ce_loss.reduction='none'
            tmp_loss = self.ce_loss(a, b)
            mask_ce_loss=(tmp_loss*self.centerness.reshape(bev_w * bev_h,1)).mean()
        else:
            mask_ce_loss = self.ce_loss(a, b)
        mask_dice_loss = self.dice_loss(pred_bev_mask.reshape(bs, -1), gt_bev_mask.reshape(bs, -1))
        return dict(mask_ce_loss=self.loss_weight*mask_ce_loss, mask_dice_loss=mask_dice_loss)

class DualFeatFusion(nn.Module):

    def __init__(self, input_channel, output_channel):
        super(DualFeatFusion, self).__init__()
        self.ca = MS_CAM(input_channel, output_channel)
    
    def forward(self, x1, x2):
        channel_factor = self.ca(torch.cat((x1,x2),1))
        out = channel_factor*x1 + (1-channel_factor)*x2

        return out

class BEVGeomAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(BEVGeomAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, bev_prob):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avg_out, max_out], dim=1)
        x1 = self.conv1(x1)
        return self.sigmoid(x1+bev_prob)

def bev_centerness_weight(nx, ny):
    xs, ys = torch.meshgrid(torch.arange(0, nx), torch.arange(0, nx))
    grid = torch.cat([xs[:, :, None], ys[:, :, None]], -1)
    grid = grid - nx//2
    grid = grid / (nx//2)
    centerness = (grid[..., 0]**2 + grid[..., 1]**2) / 2 
    centerness = centerness.sqrt() + 1
    return centerness

class DiceLoss(nn.Module):

    def __init__(self, use_sigmoid=True, loss_weight=1.):
        super(DiceLoss, self).__init__()
        self.use_sigmoid=use_sigmoid
        self.loss_weight=loss_weight

    def forward(self, inputs, targets, smooth=1e-5):
        if self.use_sigmoid:
            inputs = F.sigmoid(inputs)     
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return self.loss_weight*(1 - dice)

@NECKS.register_module()
class DualViewTransformerFull(LSSViewTransformerBEVDepth):
    def __init__(self, pc_range, bev_h=128, bev_w=128, num_height=13, collapse_z=True, loss_semantic_weight=25, 
                 depth_threshold=1, semantic_threshold=0.25, depthnet_cfg=dict(), **kwargs):
        super(DualViewTransformerFull, self).__init__(**kwargs)
        self.loss_semantic_weight = loss_semantic_weight
        self.depth_threshold = depth_threshold / self.D
        self.semantic_threshold = semantic_threshold
        self.pc_range = pc_range
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_height = num_height
        self.collapse_z = collapse_z

        self.depth_net = DepthNet(self.in_channels, self.in_channels,
                            self.out_channels, self.D+2, **depthnet_cfg)
        self.fuser = DualFeatFusion(2*self.out_channels,self.out_channels)
        self.geom_att = BEVGeomAttention()
        self.prob = ProbNet(in_channels=self.out_channels, with_centerness=True, bev_size=(self.bev_h,self.bev_w))
        self.positional_encoding = LearnedPositionalEncoding(self.out_channels // 2, self.bev_h, self.bev_w)

    def get_reference_points_3d(self, H, W, Z=8, num_points_in_pillar=13, bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in HT.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in HT, has \
                shape (bs, D, HW, 3).
        """
        zs_l = torch.linspace(3, Z-1, 5, dtype=dtype,device=device)
        zs_g = torch.linspace(0.5, Z - 0.5, num_points_in_pillar-5, dtype=dtype,device=device)
        zs = torch.cat((zs_l,zs_g)).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
        xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                            device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
        ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                            device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
        ref_3d = torch.stack((xs, ys, zs), -1)
        ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
        ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
        return ref_3d

    @force_fp32()
    def get_projection(self, rots, trans, intrins, post_rots, post_trans, bda):
        B, N, _, _ = rots.shape

        bda = bda.view(B, 1, 3, 3).repeat(1, N, 1, 1)
        inv_sensor2ego = torch.inverse(rots)
        lidar2img_R = intrins.matmul(inv_sensor2ego).matmul(torch.inverse(bda))
        lidar2img_t = -intrins.matmul(inv_sensor2ego).matmul(trans.unsqueeze(-1))
        lidar2img = torch.cat((lidar2img_R, lidar2img_t), -1)
        img_aug = torch.cat((post_rots, post_trans.unsqueeze(-1)), -1)
        return lidar2img, img_aug

    @force_fp32()
    def get_sampling_point(self, reference_points, pc_range, depth_range, lidar2img, img_aug, image_shapes):
        # B, bev_z, bev_h* bev_w, 3
        reference_points[..., 0:1] = reference_points[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]

        # B, D, HW, 3
        B, Z, num_query = reference_points.size()[:3]
        reference_points = reference_points.view(B, -1 , 3)
        num_cam = lidar2img.size(1)
        reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
        reference_points = reference_points.view(B, 1, Z*num_query, 4).repeat(1, num_cam, 1, 1)

        lidar2img = lidar2img.view(B, num_cam, 1, 3, 4)
        img_aug = img_aug.view(B, num_cam, 1, 3, 4)

        reference_points = lidar2img.matmul(reference_points.unsqueeze(-1)).squeeze(-1)

        eps = 1e-5
        referenece_depth = reference_points[..., 2:3].clone()
        bev_mask = (reference_points[..., 2:3] > eps)

        reference_points_cam = torch.cat((reference_points[..., 0:2] / torch.maximum(
            reference_points[..., 2:3], torch.ones_like(reference_points[..., 2:3])*eps), reference_points[..., 2:3], torch.ones_like(reference_points[..., 2:3])), -1)

        reference_points_cam = torch.matmul(img_aug, 
                                        reference_points_cam.unsqueeze(-1)).squeeze(-1)   
    
        reference_points_cam = reference_points_cam[..., 0:2]
        reference_points_cam[..., 0] /= image_shapes[1]
        reference_points_cam[..., 1] /= image_shapes[0]

        reference_points_cam= reference_points_cam.view(B,num_cam,Z,num_query,2)
        referenece_depth= referenece_depth.view(B,num_cam,Z,num_query,1)
        bev_mask= bev_mask.view(B,num_cam,Z,num_query,1)


        bev_mask = (bev_mask & (reference_points_cam[..., 0:1] > 0.0) 
                    & (reference_points_cam[..., 0:1] < 1.0) 
                    & (reference_points_cam[..., 1:2] > 0.0) 
                    & (reference_points_cam[..., 1:2] < 1.0))
        # D, B, N, num_query, 1
        if depth_range is not None:
            referenece_depth = (referenece_depth-depth_range[0])/(depth_range[1]-depth_range[0])
            bev_mask = (bev_mask & (referenece_depth > 0.0)
                        & (referenece_depth < 1.0))
        # [B, N, Z*Nq, 1] bev_mask
        # [B, N, Z*Nq, 2] reference_points_cam
        # [B, N, Z*Nq, 1] referenece_depth
        bev_mask = torch.nan_to_num(bev_mask)
        return torch.cat((reference_points_cam,referenece_depth),-1), bev_mask

    def init_acceleration_ht(self, coor, mask):
        """Pre-compute the necessary information in acceleration including the
        index of points in the final feature.
        """

        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.fast_sample_prepare(coor, mask)

        self.ranks_bev_ht = ranks_bev.int().contiguous()
        self.ranks_feat_ht = ranks_feat.int().contiguous()
        self.ranks_depth_ht = ranks_depth.int().contiguous()
        self.interval_starts_ht = interval_starts.int().contiguous()
        self.interval_lengths_ht = interval_lengths.int().contiguous()

    def fast_sampling(self, coor, mask, depth, feat):
        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.fast_sample_prepare(coor, mask)
        if ranks_feat is None:
            print('warning ---> no points within the predefined '
                  'bev receptive field')
            dummy = torch.zeros(size=[
                feat.shape[0], feat.shape[2],
                1, int(self.bev_h), int(self.bev_w)
            ]).to(feat)
            dummy = torch.cat(dummy.unbind(dim=2), 1)
            return dummy
        feat = feat.permute(0, 1, 3, 4, 2)
        bev_feat_shape = (depth.shape[0], 1,
                          int(self.bev_h), int(self.bev_w),
                          feat.shape[-1])  # (B, Z, Y, X, C)
        bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                               bev_feat_shape, interval_starts,
                               interval_lengths)
        # collapse Z
        if self.collapse_z:
            bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)
        return bev_feat

    def fast_sample_prepare(self, coor, mask):
        """Data preparation for voxel pooling.
        Args:
            coor (torch.tensor): Coordinate of points in the image space in
                shape (B, N, ZNq 3).
            mask (torch.tensor): mask of points in the imaage space in
                shape (B, N, ZNq, 1).
        Returns:
            tuple[torch.tensor]: Rank of the voxel that a point is belong to
                in shape (N_Points); Reserved index of points in the depth
                space in shape (N_Points). Reserved index of points in the
                feature space in shape (N_Points).
        """
        B, N, Z, Nq, _ = coor.shape
        num_points = B * N * Z * Nq
        # record the index of selected points for acceleration purpose
        ranks_bev = torch.range(
            0, num_points // (N*Z) - 1, dtype=torch.int, device=coor.device)
        ranks_bev = ranks_bev.reshape(B, 1, 1, Nq)
        ranks_bev = ranks_bev.expand(B, N, Z, Nq).flatten()
        # convert coordinate into the image feat space
        coor[..., 0] *= self.W
        coor[..., 1] *= self.H
        coor[..., 2] *= self.D
        # [B, N, Z, Nq, 3]
        coor = coor.round().long().view(num_points, 3)
        coor[..., 0].clamp_(min=0, max=self.W-1)
        coor[..., 1].clamp_(min=0, max=self.H-1)
        coor[..., 2].clamp_(min=0, max=self.D-1)
        batch_idx = torch.range(0, B*N-1).reshape(B*N, 1). \
            expand(B*N, num_points // (B*N)).reshape(num_points, 1).to(coor)
        coor = torch.cat((coor, batch_idx), 1)

        # filter out points that are outside box
        kept = mask.reshape(-1)
        if len(kept) == 0:
            return None, None, None, None, None
        coor, ranks_bev = \
            coor[kept], ranks_bev[kept]

        ranks_depth = coor[:, 3] * (self.D * self.W * self.H)
        ranks_depth += coor[:, 2] * (self.W * self.H)
        ranks_depth += coor[:, 1] * self.W + coor[:, 0]
        depth_size = B * N * self.D * self.W * self.H
        ranks_depth.clamp_(min=0, max=depth_size-1)

        ranks_feat = coor[:, 3] * (self.W * self.H)
        ranks_feat += coor[:, 1] * self.W + coor[:, 0]
        feat_size = B * N * self.W * self.H

        ranks_feat.clamp_(min=0, max=feat_size-1)

        order = ranks_bev.argsort()
        ranks_bev, ranks_depth, ranks_feat = \
            ranks_bev[order], ranks_depth[order], ranks_feat[order]

        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept)[0].int()
        if len(interval_starts) == 0:
            return None, None, None, None, None
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return ranks_bev.int().contiguous(), ranks_depth.int().contiguous(
        ), ranks_feat.int().contiguous(), interval_starts.int().contiguous(
        ), interval_lengths.int().contiguous()

    def pre_compute(self, input):
        if self.initial_flag:
            coor = self.get_lidar_coor(*input[1:7])
            self.lidar2img, self.img_aug = self.get_projection(*input[1:7])
            self.init_acceleration_v2(coor)

            self.W = self.input_size[1] / self.downsample
            self.H = self.input_size[0] / self.downsample
            voxel = self.get_reference_points_3d(self.bev_h, self.bev_w, num_points_in_pillar=self.num_height, bs=1)
            coor, mask = self.get_sampling_point(voxel, self.pc_range, self.grid_config['depth'], self.lidar2img, self.img_aug, self.input_size)
            self.init_acceleration_ht(coor, mask)

            self.initial_flag = False

    def get_lss_bev_feat(self, input, depth, tran_feat, kept=None):
        B, N, C, H, W = input[0].shape

        # Lift-Splat
        if self.accelerate:
            feat = tran_feat.view(B, N, self.out_channels, H, W)
            feat = feat.permute(0, 1, 3, 4, 2)
            depth = depth.view(B, N, self.D, H, W)
            bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                              int(self.grid_size[1]), int(self.grid_size[0]),
                              feat.shape[-1])  # (B, Z, Y, X, C)
            bev_feat = bev_pool_v2(depth, feat, self.ranks_depth,
                                   self.ranks_feat, self.ranks_bev,
                                   bev_feat_shape, self.interval_starts,
                                   self.interval_lengths)

            bev_feat = bev_feat.squeeze(2)
        else:
            coor = self.get_lidar_coor(*input[1:7])
            bev_feat = self.voxel_pooling_v2(
                coor, depth.view(B, N, self.D, H, W),
                tran_feat.view(B, N, self.out_channels, H, W))
        return bev_feat

    def get_ht_bev_feat(self, input, depth, tran_feat, bev_mask=None):
        B, N, C, H, W = input[0].shape
        self.H = H
        self.W = W

        # Prob-Sampling
        if self.accelerate:
            feat = tran_feat.view(B, N, self.out_channels, H, W)
            feat = feat.permute(0, 1, 3, 4, 2)
            depth = depth.view(B, N, self.D, H, W)
            bev_feat_shape = (depth.shape[0], 1,
                          int(self.bev_h), int(self.bev_w),
                          feat.shape[-1])  # (B, Z, Y, X, C)
            bev_feat = bev_pool_v2(depth, feat, self.ranks_depth_ht,
                                   self.ranks_feat_ht, self.ranks_bev_ht,
                                   bev_feat_shape, self.interval_starts_ht,
                                   self.interval_lengths_ht)
            if bev_mask is not None:
                bev_feat = bev_feat * bev_mask
            bev_feat = bev_feat.squeeze(2)
        else:
            lidar2img, img_aug = self.get_projection(*input[1:7])
            voxel = self.get_reference_points_3d(self.bev_h, self.bev_w, bs=B, num_points_in_pillar=self.num_height)
            coor, mask = self.get_sampling_point(voxel, self.pc_range, self.grid_config['depth'], lidar2img, img_aug, self.input_size)

            if bev_mask is not None:
                mask = bev_mask * mask.view(B,N,self.num_height,self.bev_h,self.bev_w)
            bev_feat = self.fast_sampling(
                coor, mask, depth.view(B, N, self.D, H, W),
                tran_feat.view(B, N, self.out_channels, H, W))
        return bev_feat

    def view_transform_core(self, input, depth, tran_feat):
        B, N, C, H, W = input[0].shape
        dtype = input[0].dtype

        lss_feat = self.get_lss_bev_feat(input, depth, tran_feat)
        ht_feat = self.get_ht_bev_feat(input, depth, tran_feat)

        channel_feat=self.fuser(lss_feat, ht_feat)

        mask = torch.zeros((B, self.bev_h, self.bev_w),
                    device=ht_feat.device).to(dtype)
        bev_pos = self.positional_encoding(mask).to(dtype)
        bev_mask_logit = self.prob(bev_pos + channel_feat)

        geom_feat = self.geom_att(channel_feat, bev_mask_logit) * channel_feat

        return geom_feat, depth, bev_mask_logit

    def get_downsampled_gt_depth_and_semantic(self, gt_depths, gt_semantics):
        # remove point not in depth range
        gt_semantics[gt_depths < self.grid_config['depth'][0]] = 0
        gt_semantics[gt_depths > self.grid_config['depth'][1]] = 0
        gt_depths[gt_depths < self.grid_config['depth'][0]] = 0
        gt_depths[gt_depths > self.grid_config['depth'][1]] = 0
        gt_semantic_depths = gt_depths * gt_semantics

        B, N, H, W = gt_semantics.shape
        gt_semantics = gt_semantics.view(
            B * N,
            H // self.downsample,
            self.downsample,
            W // self.downsample,
            self.downsample,
            1,
        )
        gt_semantics = gt_semantics.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_semantics = gt_semantics.view(
            -1, self.downsample * self.downsample)
        gt_semantics = torch.max(gt_semantics, dim=-1).values
        gt_semantics = gt_semantics.view(B * N, H // self.downsample,
                                   W // self.downsample)
        gt_semantics = F.one_hot(gt_semantics.long(),
                              num_classes=2).view(-1, 2).float()

        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample,
            self.downsample,
            W // self.downsample,
            self.downsample,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(
            -1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   W // self.downsample)
        gt_depths = (gt_depths -
                     (self.grid_config['depth'][0] - self.grid_config['depth'][2])) / self.grid_config['depth'][2]
        gt_depths = torch.where(
            (gt_depths < self.D + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.D + 1).view(
                                  -1, self.D + 1)[:, 1:].float()
        gt_semantic_depths = gt_semantic_depths.view(
            B * N,
            H // self.downsample,
            self.downsample,
            W // self.downsample,
            self.downsample,
            1,
        )
        gt_semantic_depths = gt_semantic_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_semantic_depths = gt_semantic_depths.view(
            -1, self.downsample * self.downsample)
        gt_semantic_depths =  torch.where(gt_semantic_depths == 0.0,
                                    1e5 * torch.ones_like(gt_semantic_depths),
                                    gt_semantic_depths)
        gt_semantic_depths = (gt_semantic_depths - (self.grid_config['depth'][0] - 
                            self.grid_config['depth'][2])) / self.grid_config['depth'][2] 
        gt_semantic_depths = torch.where(
                    (gt_semantic_depths < self.D + 1) & (gt_semantic_depths >= 0.0),
                    gt_semantic_depths, torch.zeros_like(gt_semantic_depths)).long()                           
        soft_depth_mask = gt_semantics[:,1] > 0
        gt_semantic_depths = gt_semantic_depths[soft_depth_mask]
        gt_semantic_depths_cnt = gt_semantic_depths.new_zeros([gt_semantic_depths.shape[0], self.D+1])
        for i in range(self.D+1):
            gt_semantic_depths_cnt[:,i] = (gt_semantic_depths == i).sum(dim=-1)
        gt_semantic_depths = gt_semantic_depths_cnt[:,1:] / gt_semantic_depths_cnt[:,1:].sum(dim=-1, keepdim=True)
        gt_depths[soft_depth_mask] = gt_semantic_depths

        return gt_depths, gt_semantics

    @force_fp32()
    def get_depth_and_semantic_loss(self, depth_labels, depth_preds, semantic_labels, semantic_preds):
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
        semantic_preds = semantic_preds.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        semantic_weight = torch.zeros_like(semantic_labels[:,1:2])
        semantic_weight = torch.fill_(semantic_weight, 0.1)
        semantic_weight[semantic_labels[:,1] > 0] = 0.9

        depth_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[depth_mask]
        depth_preds = depth_preds[depth_mask]
        semantic_labels = semantic_labels[depth_mask]
        semantic_preds = semantic_preds[depth_mask]
        semantic_weight = semantic_weight[depth_mask]

        with autocast(enabled=False):
            depth_loss = (F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ) * semantic_weight).sum() / max(0.1, semantic_weight.sum())

            pred = semantic_preds
            target = semantic_labels
            alpha = 0.25
            gamma = 2
            pt = (1 - pred) * target + pred * (1 - target)
            focal_weight = (alpha * target + (1 - alpha) *
                            (1 - target)) * pt.pow(gamma)
            semantic_loss = F.binary_cross_entropy(pred, target, reduction='none') * focal_weight
            semantic_loss = semantic_loss.sum() / max(1, len(semantic_loss))
        return self.loss_depth_weight * depth_loss, self.loss_semantic_weight * semantic_loss
    
    def forward(self, input, stereo_metas=None):
        (x, rots, trans, intrins, post_rots, post_trans, bda,
         mlp_input) = input[:8]

        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.depth_net(x, mlp_input)
        depth_digit = x[:, :self.D, ...]
        semantic_digit = x[:, self.D:self.D + 2]
        tran_feat = x[:, self.D + 2:self.D + 2 + self.out_channels, ...]
        depth = depth_digit.softmax(dim=1)
        semantic = semantic_digit.softmax(dim=1)
        filter_depth = torch.where(depth < self.depth_threshold, torch.zeros_like(depth), depth)
        img_mask = semantic[:,1:2] >= self.semantic_threshold
        filter_feat = img_mask*tran_feat
        bev_feat, filter_depth, bev_mask = self.view_transform(input, filter_depth, filter_feat)
        return bev_feat, depth, (bev_mask, semantic)

    def get_loss(self, depth, semantic, gt_depth, gt_semantic):
        depth_labels, semantic_labels = \
            self.get_downsampled_gt_depth_and_semantic(gt_depth, gt_semantic)
        loss_depth, loss_ce_semantic = \
            self.get_depth_and_semantic_loss(depth_labels, depth, semantic_labels, semantic)
        return loss_depth, loss_ce_semantic