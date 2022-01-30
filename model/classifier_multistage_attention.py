from torch import nn
import torch
import torch.nn.functional as F
from model.backbone.vgg import (vgg19, vgg19_bn)
from model.backbone.densenet import (densenet121, densenet169, densenet201)
from model.backbone.inception import (inception_v3)
from model.global_pool import GlobalPool
from model.attention_map import AttentionMap,CAModule


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
        self._init_attention_map_64()
        self._init_attention_map_128()
        self._init_attention_map_256()
        self._init_attention_map_512()
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
        
        self.conv64 = nn.Conv2d(64,self.backbone.num_features,1,padding=0)
        self.conv128 = nn.Conv2d(128,self.backbone.num_features,1,padding=0)
        self.conv256 = nn.Conv2d(256,self.backbone.num_features,1,padding=0)
        self.conv512 = nn.Conv2d(512,self.backbone.num_features,1,padding=0)
        
        self.pool64 = nn.AvgPool2d(8, stride=8)
        self.pool128 = nn.AvgPool2d(4, stride=4)
        self.pool256 = nn.AvgPool2d(2, stride=2)
        
        
        for index, num_class in enumerate(self.cfg.num_classes):
                setattr(self,"conv_" + str(index),nn.Conv2d(
                        self.backbone.num_features*5 ,self.backbone.num_features,kernel_size=1,stride=1,padding=0,bias=True))
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
            
    def _init_attention_map_64(self):
        if BACKBONES_TYPES[self.cfg.backbone] == 'vgg':
            setattr(self, "attention_map_64", AttentionMap(self.cfg, 512))
        elif BACKBONES_TYPES[self.cfg.backbone] == 'densenet':
            for index, num_class in enumerate(self.cfg.num_classes):
                setattr(
                    self,
                    f"attention_map_64{index}",
                    AttentionMap(
                        self.cfg,
                        64))
        elif BACKBONES_TYPES[self.cfg.backbone] == 'inception':
            setattr(self, "attention_map", AttentionMap(self.cfg, 2048))
        else:
            raise Exception(
                'Unknown backbone type : {}'.format(self.cfg.backbone)
            )
    
    def _init_attention_map_128(self):
        if BACKBONES_TYPES[self.cfg.backbone] == 'vgg':
            setattr(self, "attention_map_128", AttentionMap(self.cfg, 512))
        elif BACKBONES_TYPES[self.cfg.backbone] == 'densenet':
            for index, num_class in enumerate(self.cfg.num_classes):
                setattr(
                    self,
                    f"attention_map_128{index}",
                    AttentionMap(
                        self.cfg,
                        128))
        elif BACKBONES_TYPES[self.cfg.backbone] == 'inception':
            setattr(self, "attention_map_128", AttentionMap(self.cfg, 2048))
        else:
            raise Exception(
                'Unknown backbone type : {}'.format(self.cfg.backbone)
            )
            
    def _init_attention_map_256(self):
        if BACKBONES_TYPES[self.cfg.backbone] == 'vgg':
            setattr(self, "attention_map_256", AttentionMap(self.cfg, 512))
        elif BACKBONES_TYPES[self.cfg.backbone] == 'densenet':
            for index, num_class in enumerate(self.cfg.num_classes):
                setattr(
                    self,
                    f"attention_map_256{index}",
                    AttentionMap(
                        self.cfg,
                        256))
        elif BACKBONES_TYPES[self.cfg.backbone] == 'inception':
            setattr(self, "attention_map", AttentionMap(self.cfg, 2048))
        else:
            raise Exception(
                'Unknown backbone type : {}'.format(self.cfg.backbone)
            )      
            
    def _init_attention_map_512(self):
        if BACKBONES_TYPES[self.cfg.backbone] == 'vgg':
            setattr(self, "attention_map", AttentionMap(self.cfg, 512))
        elif BACKBONES_TYPES[self.cfg.backbone] == 'densenet':
            for index, num_class in enumerate(self.cfg.num_classes):
                setattr(
                    self,
                    f"attention_map_512{index}",
                    AttentionMap(
                        self.cfg,
                        512))
        elif BACKBONES_TYPES[self.cfg.backbone] == 'inception':
            setattr(self, "attention_map", AttentionMap(self.cfg, 2048))
        else:
            raise Exception(
                'Unknown backbone type : {}'.format(self.cfg.backbone)
            )       
    def cuda(self, device=None):
        return self._apply(lambda t: t.cuda(device))

    def forward(self, x):
        
        x0 = self.pool0(self.relu0(self.norm0(self.conv0(x))))
        x0_att = x0
        x0_maps = []
        for index, num_class in enumerate(self.cfg.num_classes):
            if self.cfg.attention_map != "None":
                
                attention_map =  getattr(self, 'attention_map_64' + str(index))
                x0_att = attention_map(x0_att)
                x0_ = self.pool64(x0_att)
                x0_ = self.conv64(x0_)
                
                x0_maps.append(x0_)
        
    
        x1 = self.dense_block1(x0_att+x0)
        x1 = self.trans_block1(x1)
        x1_att = x1
        x1_maps = []
        for index, num_class in enumerate(self.cfg.num_classes):
            if self.cfg.attention_map != "None":
                
                attention_map =  getattr(self, 'attention_map_128' + str(index))
                x1_att = attention_map(x1_att)
                
                x1_ = self.pool128(x1_att)
                x1_ = self.conv128(x1_)
                
                x1_maps.append(x1_)
        
        
        x2_maps = []
        x2 = self.trans_block2(self.dense_block2(x1+x1_att))
        x2_att = x2
        for index, num_class in enumerate(self.cfg.num_classes):
            if self.cfg.attention_map != "None":
                
                attention_map =  getattr(self, 'attention_map_256' + str(index))
                x2_att = attention_map(x2_att)
                x2_ = self.pool256(x2_att)
                x2_ = self.conv256(x2_)
                
                x2_maps.append(x2_)

        
        x3_maps = []
        
        x3 = self.trans_block3(self.dense_block3(x2+x2_att))
        x3_att = x3
        for index, num_class in enumerate(self.cfg.num_classes):
            if self.cfg.attention_map != "None":
                
                attention_map =  getattr(self, 'attention_map_512' + str(index))
                x3_att = attention_map(x3_att)
                x3_ = self.conv512(x3_att)
                x3_maps.append(x3_)
        

        feat_map  = self.dense_block4(x3+x3_att)
        
                
              
               
       
            
        
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
               
            feat_map_ = feat_map + x0_maps[index] + x1_maps[index] + x2_maps[index] + x3_maps[index]
            # conv_layer =getattr(self, 'conv_' + str(index))
            # feat_map_ = conv_layer(feat_map_)
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

        return logits, logit_maps
