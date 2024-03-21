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
import torch.nn.functional as F
from mmcv.runner import force_fp32

from mmdet.models import DETECTORS
from .. import builder
from .bevdet import BEVDet, BEVDet4D
from mmdet.models.backbones.resnet import ResNet

@DETECTORS.register_module()
class DualBEV(BEVDet):
    
    def extract_img_feat(self, img, img_metas, **kwargs):
        """Extract features of images."""
        rots, trans, intrins, post_rots, post_trans, bda = img[1:7]
        mlp_input = self.img_view_transformer.get_mlp_input(
                    rots, trans, intrins, post_rots, post_trans, bda)
        x = self.image_encoder(img[0])
        inputs=[x, rots, trans, intrins, post_rots, post_trans, bda, mlp_input]
        x, depth, bev_mask = self.img_view_transformer(inputs)
        x = self.bev_encoder(x)
        return [x], depth, bev_mask

    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        img_feats, depth, bev_mask = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = None
        return (img_feats, pts_feats, depth, bev_mask)


    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        img_feats, _, _, _ = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        assert self.with_pts_bbox
        outs = self.pts_bbox_head(img_feats)
        return outs

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats, depth, bev_mask = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)

        gt_depth = kwargs['gt_depth']
        gt_semantic = kwargs['gt_semantic']
        loss_depth, loss_ce_semantic = \
            self.img_view_transformer.get_loss(depth, bev_mask[1], gt_depth, gt_semantic)
        losses = dict(loss_depth=loss_depth, loss_ce_semantic=loss_ce_semantic)

        gt_bev_mask = kwargs['gt_bev_mask']
        loss_bev_mask = self.img_view_transformer.prob.get_bev_mask_loss(gt_bev_mask, bev_mask[0])
        losses.update(loss_bev_mask)

        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses

@DETECTORS.register_module()
class DualBEV4D(DualBEV, BEVDet4D):
    def prepare_bev_feat(self, img, rot, tran, intrin, post_rot, post_tran,
                         bda, mlp_input):
        x = self.image_encoder(img)
        inputs=[x, rot, tran, intrin, post_rot, post_tran, bda, mlp_input]
        bev_feat, depth, bev_mask = self.img_view_transformer(inputs)

        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat, depth, bev_mask

    def extract_img_feat_sequential(self, inputs, feat_prev):
        imgs, rots_curr, trans_curr, intrins = inputs[:4]
        rots_prev, trans_prev, post_rots, post_trans, bda = inputs[4:]
        bev_feat_list = []
        mlp_input = self.img_view_transformer.get_mlp_input(
            rots_curr[0:1, ...], trans_curr[0:1, ...], intrins, post_rots,
            post_trans, bda[0:1, ...])
        inputs_curr = (imgs, rots_curr[0:1, ...], trans_curr[0:1, ...],
                       intrins, post_rots, post_trans, bda[0:1,
                                                           ...], mlp_input)
        bev_feat, depth, bev_mask = self.prepare_bev_feat(*inputs_curr)
        bev_feat_list.append(bev_feat)

        # align the feat_prev
        _, C, H, W = feat_prev.shape
        feat_prev = \
            self.shift_feature(feat_prev,
                               [trans_curr, trans_prev],
                               [rots_curr, rots_prev],
                               bda)
        bev_feat_list.append(feat_prev.view(1, (self.num_frame - 1) * C, H, W))

        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        return [x], depth, bev_mask

    def extract_img_feat(self,
                         img,
                         img_metas,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        if sequential:
            return self.extract_img_feat_sequential(img, **kwargs)
        imgs, rots, trans, intrins, post_rots, post_trans, bda = \
            self.prepare_inputs(img)
        """Extract features of images."""
        bev_feat_list = []
        depth_list = []
        bev_mask_list = []
        key_frame = True  # back propagation for key frame only
        for img, rot, tran, intrin, post_rot, post_tran in zip(
                imgs, rots, trans, intrins, post_rots, post_trans):
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    rot, tran = rots[0], trans[0]
                mlp_input = self.img_view_transformer.get_mlp_input(
                    rots[0], trans[0], intrin, post_rot, post_tran, bda)
                inputs_curr = (img, rot, tran, intrin, post_rot,
                               post_tran, bda, mlp_input)
                if key_frame:
                    bev_feat, depth, bev_mask = self.prepare_bev_feat(*inputs_curr)
                else:
                    with torch.no_grad():
                        bev_feat, depth, bev_mask = self.prepare_bev_feat(*inputs_curr)
            else:
                bev_feat = torch.zeros_like(bev_feat_list[0])
                depth = None
                bev_mask = None
            bev_feat_list.append(bev_feat)
            depth_list.append(depth)
            bev_mask_list.append(bev_mask)
            key_frame = False
        if pred_prev:
            assert self.align_after_view_transfromation
            assert rots[0].shape[0] == 1
            feat_prev = torch.cat(bev_feat_list[1:], dim=0)
            trans_curr = trans[0].repeat(self.num_frame - 1, 1, 1)
            rots_curr = rots[0].repeat(self.num_frame - 1, 1, 1, 1)
            trans_prev = torch.cat(trans[1:], dim=0)
            rots_prev = torch.cat(rots[1:], dim=0)
            bda_curr = bda.repeat(self.num_frame - 1, 1, 1)
            return feat_prev, [
                imgs[0], rots_curr, trans_curr, intrins[0], rots_prev,
                trans_prev, post_rots[0], post_trans[0], bda_curr
            ]
        if self.align_after_view_transfromation:
            for adj_id in range(1, self.num_frame):
                bev_feat_list[adj_id] = \
                    self.shift_feature(bev_feat_list[adj_id],
                                       [trans[0], trans[adj_id]],
                                       [rots[0], rots[adj_id]],
                                       bda)
        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        return [x], depth_list[0], bev_mask_list[0]

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats, depth, bev_mask = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)

        gt_depth = kwargs['gt_depth']
        gt_semantic = kwargs['gt_semantic']
        loss_depth, loss_ce_semantic = \
            self.img_view_transformer.get_loss(depth, bev_mask[1], gt_depth, gt_semantic)
        losses = dict(loss_depth=loss_depth, loss_ce_semantic=loss_ce_semantic)

        gt_bev_mask = kwargs['gt_bev_mask']
        loss_bev_mask = self.img_view_transformer.prob.get_bev_mask_loss(gt_bev_mask, bev_mask[0])
        losses.update(loss_bev_mask)

        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses