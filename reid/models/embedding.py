import math
import copy
from torch import nn
import torch
import torch.nn.functional as F
from torch import nn


class VNetEmbed(nn.Module):
    def __init__(self, instances_num=4, feat_num=2048, num_classes=0, drop_ratio=0.5):
        super(VNetEmbed, self).__init__()
        self.instances_num = instances_num
        self.feat_num =2048 #3072
        self.temp = 1
        #self.kron = KronMatching()
        self.bn = nn.BatchNorm1d(self.feat_num)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()
        self.classifier = nn.Linear(self.feat_num, num_classes)
        self.classifier.weight.data.normal_(0, 0.001)
        self.classifier.bias.data.zero_()
        self.drop = nn.Dropout(drop_ratio)

        self.classifier2 = nn.Linear(self.feat_num, 1)
        self.classifier2.weight.data.normal_(0, 0.001)
        self.classifier2.bias.data.zero_()

    

    def forward(self, probe_x, gallery_x, p2g=True, g2g=False):
        if not self.training and len(probe_x.size()) != len(gallery_x.size()):
            probe_x = probe_x.unsqueeze(0)

        probe_x.contiguous()
        gallery_x.contiguous()

        if g2g is True:            
            N_probe, C, H, W = probe_x.size()
            N_probe = probe_x.size(0)
            N_gallery = gallery_x.size(0)
    
            probe_x = F.avg_pool2d(probe_x, probe_x.size()[2:]).view(N_probe,
                                                                     C)  # average pooling the prob_x
            gallery_x = F.avg_pool2d(gallery_x, gallery_x.size()[2:]).view(N_gallery,
                                                                           C)  # average pooling the gallary_x
            probe_x = probe_x.unsqueeze(1)
            f_len = gallery_x.size(1)
            pro = probe_x.view(N_probe, f_len).contiguous()
            pro = self.bn(pro)
    
            galle = probe_x.view(N_probe, f_len).contiguous()
            galle = self.bn(galle)
    
            dot_p = torch.matmul(pro, galle.transpose(1, 0)).contiguous()
            dot_p = dot_p.view(dot_p.size(0), dot_p.size(0))
            #dot_p = torch.pow(dot_p, 2)
    
            probe_x = probe_x.expand(N_probe, N_gallery,
                                     self.feat_num)  # reshaping prob_x to a square matrix, (numb_gallery image X number_gallery_image)
            probe_x = probe_x.contiguous()
            gallery_x = gallery_x.unsqueeze(0)
            gallery_x = gallery_x.expand(N_probe, N_gallery,
                                         self.feat_num)  # reshaping gallary_x to a square matrix, (numb_gallery image X number_gallery_image)
    
    
    
            #probe_x = self._kron_matching(gallery_x, probe_x)
    
    
            gallery_x = gallery_x.contiguous()
            diff = gallery_x - probe_x  # Computing the distance

        diff = torch.pow(diff, 2)
        #dot_p = torch.pow(dot_p,2)

        #diff= torch.exp(diff) # using exponent instead of square doesn't work

        diff = diff.view(N_probe * N_gallery, -1)
        diff = diff.contiguous()
        bn_diff = self.bn(diff)
        bn_diff = self.drop(bn_diff)

        cls_encode = self.classifier(bn_diff)
        # cls_encode = torch.mean(bn_diff
        cls_encode = cls_encode.view(N_probe, N_gallery, -1)
        #dot_p = self.classifier2(bn_diff)
        #dot_p = dot_p.view(N_probe, N_gallery)

        return cls_encode, dot_p






