from torch import nn

import torch.nn.functional as F
from model.backbone.vgg import (vgg19, vgg19_bn)
from model.backbone.densenet import (densenet121, densenet169, densenet201)
from model.backbone.inception import (inception_v3)
from model.global_pool import GlobalPool
from model.attention_map import AttentionMap


BACKBONES = {'vgg19': vgg19,
             'vgg19_bn': vgg19_bn,
             'densenet121': densenet121,
             'densenet169': densenet169,
             'densenet201': densenet201,
             'inception_v3': inception_v3}


BACKBONES_TYPES = {'vgg19': 'vgg',
                   'vgg19_bn': 'vgg',
                   'densenet121': 'densenet',
                   'densenet169': 'densenet',
                   'densenet201': 'densenet',
                   'inception_v3': 'inception'}
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import efficientnet_pytorch
from efficientnet_pytorch import EfficientNet

__all__ = ['EfficientNet', 'EfficientNet_AG']

def my_efficientnet_forward(self, inputs):
    """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
    bs = inputs.size(0)
    
    # Convolution layers
    x = self.extract_features(inputs)
    
    # Pooling and final linear layer
    #print(x.shape)
    
    return x

efficientnet_pytorch.EfficientNet.forward.__code__ = my_efficientnet_forward.__code__

def EfficientNet_AG(compound_coeff=0, num_classes= 1000, **kwargs):
  
    model = EfficientNet_(num_init_features=64, compound_coeff=compound_coeff, num_classes=num_classes )
    return model

class EfficientNet_(nn.Module) :
    
    def __init__(self,num_init_features=3, compound_coeff = 0, weights_path=None, num_classes=5) :
        super(EfficientNet_, self).__init__()
        
        model= EfficientNet.from_pretrained(f'efficientnet-b{compound_coeff}',in_channels=3)
        
        self.model = model
        self.model.add_module('norm5', nn.BatchNorm2d(1536))
        self.classifier = nn.Linear(1536, num_classes)
    
        self.Sigmoid = nn.Sigmoid()
    
    
    def forward(self, x):
#        features = self.features(x)
#        print(features.shape)
        features = self.model(x)
#        features = self.bn(features) 
        #print(features.shape)
        # out = F.relu(features, inplace=True)
        # out_after_pooling = F.avg_pool2d(out, kernel_size=7, stride=7).view(features.size(0), -1)
        # #print(out_after_pooling.shape)
        # out = self.classifier(out_after_pooling)
        # out = self.Sigmoid(out)
        # #print(out.shape)                   
        return features

class Classifier(nn.Module):

    def __init__(self, cfg):
        super(Classifier, self).__init__()
        self.cfg = cfg
        self.backbone = BACKBONES[cfg.backbone](cfg)
        self.global_pool = GlobalPool(cfg)
        self.expand = 1
        if cfg.global_pool == 'AVG_MAX':
            self.expand = 2
        elif cfg.global_pool == 'AVG_MAX_LSE':
            self.expand = 3
        self._init_classifier()
        self._init_bn()
        self._init_attention_map()
        self.efficient_net = EfficientNet_AG(3)
    def _init_classifier(self):
        for index, num_class in enumerate(self.cfg.num_classes):
            if BACKBONES_TYPES[self.cfg.backbone] == 'vgg':
                setattr(
                    self,
                    "fc_" + str(index),
                    nn.Conv2d(
                        512 * self.expand,
                        num_class,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True))
            elif BACKBONES_TYPES[self.cfg.backbone] == 'densenet':
                setattr(
                    self,
                    "fc_" +
                    str(index),
                    nn.Conv2d(
                        1536 *
                        self.expand,
                        num_class,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True))
            elif BACKBONES_TYPES[self.cfg.backbone] == 'inception':
                setattr(
                    self,
                    "fc_" + str(index),
                    nn.Conv2d(
                        2048 * self.expand,
                        num_class,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True))
            else:
                raise Exception(
                    'Unknown backbone type : {}'.format(self.cfg.backbone)
                )

            classifier = getattr(self, "fc_" + str(index))
            if isinstance(classifier, nn.Conv2d):
                classifier.weight.data.normal_(0, 0.01)
                classifier.bias.data.zero_()

    def _init_bn(self):
        for index, num_class in enumerate(self.cfg.num_classes):
            if BACKBONES_TYPES[self.cfg.backbone] == 'vgg':
                setattr(self, "bn_" + str(index),
                        nn.BatchNorm2d(512 * self.expand))
            elif BACKBONES_TYPES[self.cfg.backbone] == 'densenet':
                setattr(
                    self,
                    "bn_" +
                    str(index),
                    nn.BatchNorm2d(
                        self.backbone.num_features *
                        self.expand))
            elif BACKBONES_TYPES[self.cfg.backbone] == 'inception':
                setattr(self, "bn_" + str(index),
                        nn.BatchNorm2d(2048 * self.expand))
            else:
                raise Exception(
                    'Unknown backbone type : {}'.format(self.cfg.backbone)
                )

    def _init_attention_map(self):
        if BACKBONES_TYPES[self.cfg.backbone] == 'vgg':
            setattr(self, "attention_map", AttentionMap(self.cfg, 512))
        elif BACKBONES_TYPES[self.cfg.backbone] == 'densenet':
            for index, num_class in enumerate(self.cfg.num_classes):
                setattr(
                    self,
                    f"attention_map_{index}",
                    AttentionMap(
                        self.cfg,
                        1536))
        elif BACKBONES_TYPES[self.cfg.backbone] == 'inception':
            setattr(self, "attention_map", AttentionMap(self.cfg, 2048))
        else:
            raise Exception(
                'Unknown backbone type : {}'.format(self.cfg.backbone)
            )

    def cuda(self, device=None):
        return self._apply(lambda t: t.cuda(device))

    def forward(self, x):
        # (N, C, H, W)
        feat_map = self.efficient_net(x)
        #print(feat_map.shape)
        # [(N, 1), (N,1),...]
        logits = list()
        # [(N, H, W), (N, H, W),...]
        logit_maps = list()
        for index, num_class in enumerate(self.cfg.num_classes):
            if self.cfg.attention_map != "None":
                
                attention_map =  getattr(self, 'attention_map_' + str(index))
                feat_map = attention_map(feat_map)
            
            classifier = getattr(self, "fc_" + str(index))
            # (N, 1, H, W)
            logit_map = None
            if not (self.cfg.global_pool == 'AVG_MAX' or
                    self.cfg.global_pool == 'AVG_MAX_LSE'):
                logit_map = classifier(feat_map)
                logit_maps.append(logit_map.squeeze())
               
            # (N, C, 1, 1)
            feat = self.global_pool(feat_map, logit_map)
                
            if self.cfg.fc_bn:
                bn = getattr(self, "bn_" + str(index))
                feat = bn(feat)
            feat = F.dropout(feat, p=self.cfg.fc_drop, training=self.training)
            # (N, num_class, 1, 1)

            logit = classifier(feat)
           
            # (N, num_class)
            logit = logit.squeeze(-1).squeeze(-1)
            logits.append(logit)

        return (logits, logit_maps)
