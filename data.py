"""
    PartNetShapeDataset
"""

import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import json
from utils import load_pts


class PartNetShapeDataset(data.Dataset):

    def __init__(self, category, split, mode, data_features, \
            img_size=224, num_view=24, num_point=10000):
        # store parameters
        self.pts_data_dir = './data/%s/pts_data/' % category
        self.img_data_dir = './data/%s/render_data/' % category

        self.category = category
        self.split = split
        self.mode = mode    # camera_space or shape_space

        self.img_size = img_size
        self.num_view = num_view
        self.num_point = num_point

        # load data
        with open(os.path.join('./stats', category+'_'+split+'.txt'), 'r') as fin:
            self.data = [l.rstrip() for l in fin.readlines()]

        # load cam matrixs
        with open(os.path.join('./data', category, 'cameras.json'), 'r') as fin:
            self.cam_mats = json.load(fin)

        # data features
        self.data_features = data_features

    def __str__(self):
        strout = '[PartNetShapeDataset %s %s %d %s]' % (self.category, self.split, len(self), self.mode)
        return strout

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        view_id = np.random.randint(self.num_view)

        data_feats = ()
        for feat in self.data_features:
            if feat == 'img':
                with Image.open(os.path.join(self.img_data_dir, self.data[index], 'view-%02d'%view_id, 'shape-rgb.png')) as fimg:
                    out = np.array(fimg, dtype=np.float32) / 255
                white_img = np.ones((self.img_size, self.img_size, 3), dtype=np.float32)
                mask = np.tile(out[:, :, 3:4], [1, 1, 3])
                out = out[:, :, :3] * mask + white_img * (1 - mask)
                out = torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)
                data_feats = data_feats + (out,)
            
            elif feat == 'pts':
                out = load_pts(os.path.join(self.pts_data_dir, self.data[index], 'point_sample', 'sample-points-all-pts-nor-rgba-10000.txt'))[:self.num_point]

                if self.mode == 'camera_space':
                    d = np.array([float(i) for i in self.cam_mats['%s-%d'%(self.data[index], view_id)].split(';')], dtype=np.float32).reshape(3, 4)
                    trans_shape_to_camera2 = np.eye(4).astype(np.float32)
                    trans_shape_to_camera2[:3, :] = d

                    trans_camera2_to_camera = np.eye(4).astype(np.float32)
                    trans_camera2_to_camera[0, 3] = -2

                    view_trans = np.dot(trans_camera2_to_camera, trans_shape_to_camera2)
                    scale = np.linalg.norm(view_trans[:3, 0])
                    view_trans[:3, :] /= scale

                    out = np.dot(out, view_trans[:3, :3].T) + np.tile(view_trans[:3, 3].reshape(1, 3), [self.num_point, 1])

                out = torch.from_numpy(out).float().unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'shape_id':
                data_feats = data_feats + (self.data[index],)

            elif feat == 'view_id':
                data_feats = data_feats + (view_id,)

            else:
                raise ValueError('ERROR: unknown feat type %s!' % feat)

        return data_feats

