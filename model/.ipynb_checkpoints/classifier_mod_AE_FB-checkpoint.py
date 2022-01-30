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

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
       
        #Encoder
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1) #320-->160 
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)#160-->80
        self.conv3 = nn.Conv2d(64, 256, 3, padding=1)#80-->40
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)#40-->20
        self.conv5 = nn.Conv2d(512, 1024, 3, padding=1)#20-->10
        self.pool = nn.MaxPool2d(2, 2)
       
        #Decoder
        self.t_conv1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(256, 64, 2, stride=2)
        self.t_conv4 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.t_conv5 = nn.ConvTranspose2d(64, 3, 2, stride=2)
        


    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x_b = self.pool(F.relu(self.conv5(x)))
        
        x = F.relu(self.t_conv1(x_b))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        x = F.relu(self.t_conv4(x))
        x = F.relu(self.t_conv5(x))
        

        
              
        return x, x_b

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
        self.conv0 = self.backbone.features.conv0
        self.norm0 = self.backbone.features.norm0
        self.relu0 = self.backbone.features.relu0
        self.pool0 = self.backbone.features.pool0

        ############# Block1-down 64-64  ##############
        self.dense_block1 = self.backbone.features.denseblock1
        self.trans_block1 = self.backbone.features.transition1

        ############# Block2-down 32-32  ##############
        self.dense_block2 = self.backbone.features.denseblock2
        self.trans_block2 = self.backbone.features.transition2

        ############# Block3-down  16-16 ##############
        self.dense_block3 = self.backbone.features.denseblock3
        self.trans_block3 = self.backbone.features.transition3
        
        self.dense_block4 = self.backbone.features.denseblock4
       
        
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
                        self.backbone.num_features *
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
                        self.backbone.num_features))
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
        x0 = self.pool0(self.relu0(self.norm0(self.conv0(x))))
        print(x0.shape)
        ## 64 X 64
        x1 = self.dense_block1(x0)
        x1 = self.trans_block1(x1)
        print(x1.shape)
        ###  32x32
        x2 = self.trans_block2(self.dense_block2(x1))
        print(x2.shape)
        ### 16 X 16
        x3 = self.trans_block3(self.dense_block3(x2))
        print(x3.shape)
        ## 8 X 8
        feat_map = self.dense_block4(x3)
        
        #feat_map = self.backbone(x)
        
        #
        # feat_map = x
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

        return logits, logit_maps, feat_map
