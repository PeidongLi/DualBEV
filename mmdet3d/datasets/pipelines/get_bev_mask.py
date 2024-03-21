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

import random
import warnings

import cv2
import torch
import numpy as np

from ..builder import OBJECTSAMPLERS, PIPELINES


def rotz(t):
    """Rotation about the z-axis.
    :param t: rotation angle
    :return: rotation matrix
    """

    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s],
                     [s, c]])

def get_corners_2d(box):
    """Takes an bounding box and calculate the 2D corners in BEV plane.
    0 --- 1
    |     |        x
    |     |        ^
    |     |        |
    3 --- 2  y <---o
    :param box: 3D bounding box, [x, y, z, l, w, h, r]
    :return: corners_2d: (4,2) array in left image coord.
    """

    # compute rotational matrix around yaw axis
    rz = box[6]
    R = rotz(rz)

    # 2d bounding box dimensions
    l = box[3]
    w = box[4]

    # 2d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_2d = np.dot(R, np.vstack([x_corners, y_corners]))
    corners_2d[0, :] = corners_2d[0, :] + box[0]
    corners_2d[1, :] = corners_2d[1, :] + box[1]

    return corners_2d.T


@PIPELINES.register_module()
class GetBEVMask(object):
    """
    """

    def __init__(self,
                 point_cloud_range=[-50, -51.2, -2, 154.8, 51.2, 6],
                 voxel_size=[0.32, 0.32, 8],
                 downsample_ratio=2):
        self.pointcloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.downsample_ratio = downsample_ratio

    def plot_gt_boxes_mask(self, gt_boxes):
        """ Visualize the ground truth boxes.
        :param gt_boxes: gt boxes, [N, [x, y, z, l, w, h, r]]
        :param bev_range: bev range, [x_min, y_min, z_min, x_max, y_max, z_max]
        :return: None
        """
        # Configure the resolution
        step_x = self.voxel_size[0]
        step_y = self.voxel_size[1]
        # Initialize the plotting canvas
        bev_range = self.pointcloud_range
        pixels_x = int((bev_range[3] - bev_range[0]) / step_x)
        pixels_y = int((bev_range[4] - bev_range[1]) / step_y)
        canvas = np.zeros((pixels_x, pixels_y), np.uint8)
        canvas.fill(0)
        bev_height = np.zeros((pixels_x, pixels_y), np.float32)

        for box in gt_boxes:
            box2d = get_corners_2d(box)
            box2d[:, 0] -= bev_range[0]
            box2d[:, 1] -= bev_range[1]
            box2d[:, 0] /= step_x
            box2d[:, 1] /= step_y
            temp = box2d[:, 0].copy()
            box2d[:,0] = box2d[:,1]
            box2d[:,1] = temp
            box2d = box2d.astype(np.int32)
            cv2.fillPoly(img=canvas, pts=[box2d], color=1, lineType=cv2.LINE_AA)

        canvas = cv2.flip(canvas, 0)
        canvas = cv2.flip(canvas, 1)

        canvas = cv2.rotate(canvas, cv2.ROTATE_90_CLOCKWISE)
        canvas = cv2.flip(canvas, 0)
        # canvas = cv2.resize(canvas, dsize=(pixels_x//self.downsample_ratio, pixels_y//self.downsample_ratio))
        canvas = torch.from_numpy(canvas).long()

        return canvas


    def __call__(self, input_dict):
        """Private function to rotate, scale and translate bounding boxes and
        points.
        Args:
            input_dict (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
                'pcd_scale_factor', 'pcd_trans' and keys in
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        gt_boxes_3d = input_dict['gt_bboxes_3d'].tensor.cpu().numpy()
        gt_masks = self.plot_gt_boxes_mask(gt_boxes=gt_boxes_3d)
        input_dict["gt_bev_mask"] = gt_masks
        return input_dict


    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(rot_range={self.rot_range},'
        repr_str += f' scale_ratio_range={self.scale_ratio_range},'
        repr_str += f' translation_std={self.translation_std},'
        repr_str += f' shift_height={self.shift_height})'
        return repr_str