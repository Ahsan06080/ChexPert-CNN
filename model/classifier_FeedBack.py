from torch import nn
import torch
import torch.nn.functional as F
from model.backbone.vgg import (vgg19, vgg19_bn)
from model.backbone.densenet import (densenet121, densenet169, densenet201)
from model.backbone.inception import (inception_v3)
from model.global_pool import GlobalPool
from model.attention_map import AttentionMap,


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
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)#80-->40
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)#40-->20
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)#20-->10
        self.conv6 = nn.Conv2d(512, 1024, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        
        self.upsample3  = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample2  = nn.Upsample(scale_factor=4, mode='nearest')
        self.upsample1  = nn.Upsample(scale_factor=8, mode='nearest')
        
        self.conv4_1x1 = nn.Conv2d(1024, 512, 1)#10
        self.conv3_1x1 = nn.Conv2d(1024, 256, 1)#20
        self.conv2_1x1 = nn.Conv2d(1024, 128, 1)#40
        self.conv1_1x1 = nn.Conv2d(1024, 64, 1)#80
        #Decoder
       
        


    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x_b = F.relu(self.conv6(x))
        x_4 = F.relu(self.conv4_1x1(x_b))
        x_3 = self.upsample3(x_b)
        x_3 = F.relu(self.conv3_1x1(x_3))
        x_2 = self.upsample2(x_b)
        x_2 = F.relu(self.conv2_1x1(x_2))
        x_1 = self.upsample1(x_b)
        x_1 = F.relu(self.conv1_1x1(x_1))
    
        
        

        
              
        return [x_1,x_2,x_3,x_4], x_b

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
        
        self.conv64 = nn.Conv2d(128,64,1)
        self.conv128 = nn.Conv2d(256,128,1)
        self.conv256 = nn.Conv2d(512,256,1)
        self.conv512 = nn.Conv2d(1024,512,1)
        
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

        ## 64 X 64
        x1 = self.dense_block1(x0)
        x1 = self.trans_block1(x1)

        ###  32x32
        x2 = self.trans_block2(self.dense_block2(x1))

        ### 16 X 16
        x3 = self.trans_block3(self.dense_block3(x2))


        ## 8 X 8
        feat_map  = self.dense_block4(x3)
                
              
               
       
            
        
        #
        # feat_map = x
        # [(N, 1), (N,1),...]
        logits = list()
        # [(N, H, W), (N, H, W),...]
        logit_maps = 0
        for index, num_class in enumerate(self.cfg.num_classes):
            if self.cfg.attention_map != "None":
                
                attention_map =  getattr(self, 'attention_map_' + str(index))
                feat_map = attention_map(feat_map)
                logit_maps +=feat_map
            classifier = getattr(self, "fc_" + str(index))
            # (N, 1, H, W)
            
            
            logit_map = None
            
               
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

        return logits, logit_maps
    
class Classifier2(nn.Module):

    def __init__(self, cfg):
        super(Classifier2, self).__init__()
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
        
        self.upsample3  = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample2  = nn.Upsample(scale_factor=4, mode='nearest')
        self.upsample1  = nn.Upsample(scale_factor=8, mode='nearest')
        
        self.conv64 = nn.Conv2d(1024,64,1)
        self.conv128 = nn.Conv2d(1024,128,1)
        self.conv256 = nn.Conv2d(1024,256,1)
        self.conv512 = nn.Conv2d(1024,512,1)
        
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

    def forward(self, x,x_FE):
        # (N, C, H, W)
        x0 = self.pool0(self.relu0(self.norm0(self.conv0(x))))
        x0_FE = self.upsample1(x_FE)
        x0_FE = self.conv64(x0_FE)
        x0 = x0+x0_FE
        ## 64 X 64
        
        x1 = self.dense_block1(x0)
        x1 = self.trans_block1(x1)
        x1_FE = self.upsample2(x_FE)
        x1_FE = self.conv128(x1_FE)
        x1= x1+x1_FE

        ###  32x32
        x2 = self.trans_block2(self.dense_block2(x1))
        x2_FE = self.upsample3(x_FE)
        x2_FE = self.conv256(x2_FE)
        x2 = x2+x2_FE
        ### 16 X 16
        x3 = self.trans_block3(self.dense_block3(x2))
        x3_FE = self.conv512(x_FE)
        x3 = x3+x3_FE


        ## 8 X 8
        feat_map  = self.dense_block4(x3)
                
              
               
       
            
        
        #
        # feat_map = x
        # [(N, 1), (N,1),...]
        logits = list()
        # [(N, H, W), (N, H, W),...]
        logit_maps = 0
        for index, num_class in enumerate(self.cfg.num_classes):
            if self.cfg.attention_map != "None":
                
                attention_map =  getattr(self, 'attention_map_' + str(index))
                feat_map = attention_map(feat_map)
                logit_maps +=feat_map
            classifier = getattr(self, "fc_" + str(index))
            # (N, 1, H, W)
            
            
            logit_map = None
            
               
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

        return logits, logit_maps