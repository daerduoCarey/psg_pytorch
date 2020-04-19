"""
    Resnet50 + PSG (two modes: fc or fc_upconv)
"""

import torch
from torch import nn
import torch.nn.functional as F
import sys, os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from torchvision.models import resnet50
sys.path.append(os.path.join(BASE_DIR, '..'))
from cd.chamfer import chamfer_distance


class FCDecoder(nn.Module):

    def __init__(self, num_point=2048):
        super(FCDecoder, self).__init__()
        print('Using FCDecoder!')

        self.mlp1 = nn.Linear(1024, 1024)
        self.mlp2 = nn.Linear(1024, 1024)
        self.mlp3 = nn.Linear(1024, num_point*3)

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(1024)

    def forward(self, feat):
        batch_size = feat.shape[0]

        net = feat
        net = torch.relu(self.bn1(self.mlp1(net)))
        net = torch.relu(self.bn2(self.mlp2(net)))
        net = self.mlp3(net).view(batch_size, -1, 3)

        return net


class FCUpconvDecoder(nn.Module):

    def __init__(self, num_point=2048):
        super(FCUpconvDecoder, self).__init__()
        print('Using FCUpconvDecoder!')

        self.mlp1 = nn.Linear(1024, 1024)
        self.mlp2 = nn.Linear(1024, 1024)
        self.mlp3 = nn.Linear(1024, 1024*3)

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(1024)

        self.deconv1 = nn.ConvTranspose2d(1024, 1024, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, 3, 1)
        self.deconv3 = nn.ConvTranspose2d(512, 256, 4, 2)
        self.deconv4 = nn.ConvTranspose2d(256, 128, 5, 3)
        self.deconv5 = nn.ConvTranspose2d(128, 3, 1, 1)

        self.debn1 = nn.BatchNorm2d(1024)
        self.debn2 = nn.BatchNorm2d(512)
        self.debn3 = nn.BatchNorm2d(256)
        self.debn4 = nn.BatchNorm2d(128)

    def forward(self, feat):
        batch_size = feat.shape[0]

        fc_net = feat
        fc_net = torch.relu(self.bn1(self.mlp1(fc_net)))
        fc_net = torch.relu(self.bn2(self.mlp2(fc_net)))
        fc_net = self.mlp3(fc_net).view(batch_size, -1, 3)

        upconv_net = feat.view(batch_size, -1, 1, 1)
        upconv_net = torch.relu(self.debn1(self.deconv1(upconv_net)))
        upconv_net = torch.relu(self.debn2(self.deconv2(upconv_net)))
        upconv_net = torch.relu(self.debn3(self.deconv3(upconv_net)))
        upconv_net = torch.relu(self.debn4(self.deconv4(upconv_net)))
        upconv_net = self.deconv5(upconv_net).view(batch_size, 3, -1).permute(0, 2, 1)

        net = torch.cat([fc_net, upconv_net], dim=1)

        return net


class Network(nn.Module):

    def __init__(self, conf):
        super(Network, self).__init__()
        self.conf = conf

        self.resnet = resnet50(pretrained=conf.pretrain_resnet)
        self.resnet.fc = nn.Linear(in_features=2048, out_features=1024, bias=True)
        self.resnet_final_bn = nn.BatchNorm1d(1024)

        if conf.decoder_type == 'fc':
            self.decoder = FCDecoder()
        elif conf.decoder_type == 'fc_upconv':
            self.decoder = FCUpconvDecoder()
        else:
            raise ValueError('ERROR: unknown decoder_type %s!' % conf.decoder_type)
    
    def forward(self, img):
        img = img.clone()
        img[:, 0] = (img[:, 0] - 0.485) / 0.229
        img[:, 1] = (img[:, 1] - 0.456) / 0.224
        img[:, 2] = (img[:, 2] - 0.406) / 0.225
        
        img_feat = torch.relu(self.resnet_final_bn(self.resnet(img)))
        
        pc = self.decoder(img_feat)
        
        return pc

    def get_pc_loss(self, pc1, pc2):
        dist1, dist2 = chamfer_distance(pc1, pc2, transpose=False)
        loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
        return loss_per_data * 100
 
