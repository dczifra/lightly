import torch
import torchvision
import torch.nn as nn

import lightly.data as ldata
import lightly.models as models
import lightly.loss as loss
import pytorch_lightning as pl
from lightly.utils import BenchmarkModule
from lightly.models.resnet import ResNetGenerator

from lightly.loss import NegativeCosineSimilarity, NTXentLoss, SwaVLoss, TsLoss, TwistLoss
from lightly.models.modules.heads import SwaVProjectionHead, SwaVPrototypes, SimCLRProjectionHead, SimSiamPredictionHead, ProjectionHead

gather_distributed = False 
# ========================================
#                 MODELS
# ========================================
class SimCLRModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes, lr_factor, max_epochs):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        #resnet = torchvision.models.resnet18()
        resnet = ResNetGenerator('resnet-18')
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1)
        )
        self.projection_head = SimCLRProjectionHead(512, 512, 128)
        self.criterion = NTXentLoss()

        #self.dummy_param.device = 'cuda:0'
        self.lr_factor = lr_factor
        self.max_epochs = max_epochs
    
    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), 
            lr=6e-2 * self.lr_factor,
            momentum=0.9, 
            weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]


class SwaVModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes, lr_factor, max_epochs):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = ResNetGenerator('resnet-18')
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1)
        )

        self.projection_head = SwaVProjectionHead(512, 512, 128)
        self.prototypes = SwaVPrototypes(128, 512) # use 512 prototypes

        self.criterion = SwaVLoss(sinkhorn_gather_distributed=gather_distributed)

        self.lr_factor = lr_factor
        self.max_epochs = max_epochs

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x)
        x = nn.functional.normalize(x, dim=1, p=2)
        return self.prototypes(x)

    def training_step(self, batch, batch_idx):
        # normalize the prototypes so they are on the unit sphere
        self.prototypes.normalize()

        # the multi-crop dataloader returns a list of image crops where the
        # first two items are the high resolution crops and the rest are low
        # resolution crops
        multi_crops, _, _ = batch
        multi_crop_features = [self.forward(x) for x in multi_crops]

        # split list of crop features into high and low resolution
        high_resolution_features = multi_crop_features[:2]
        low_resolution_features = multi_crop_features[2:]

        # calculate the SwaV loss
        loss = self.criterion(
            high_resolution_features,
            low_resolution_features
        )

        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(),
            lr=1e-3 * self.lr_factor,
            weight_decay=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]


class SimSiamModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes, max_epochs):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = ResNetGenerator('resnet-18')
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1)
        )
        self.prediction_head = SimSiamPredictionHead(2048, 512, 2048)
        # use a 2-layer projection head for cifar10 as described in the paper
        self.projection_head = ProjectionHead([
            (
                512,
                2048,
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True)
            ),
            (
                2048,
                2048,
                nn.BatchNorm1d(2048),
                None
            )
        ])
        self.criterion = NegativeCosineSimilarity()

        self.max_epochs = max_epochs
            
    def forward(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0, p0 = self.forward(x0)
        z1, p1 = self.forward(x1)
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), 
            lr=6e-2, # no lr-scaling, results in better training stability
            momentum=0.9,
            weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]


class SwaV_ts_Model(BenchmarkModule):
    def __init__(self, dataloader_kNN, dataloader_prototype, num_classes, lr_factor, max_epochs):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = ResNetGenerator('resnet-18')
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1)
        )

        self.projection_head = SwaVProjectionHead(512, 512, 128)
        self.prototypes = SwaVPrototypes(128, 512) # use 512 prototypes

        self.criterion = SwaVLoss(sinkhorn_gather_distributed=gather_distributed)

        self.lr_factor = lr_factor
        self.max_epochs = max_epochs

        self.dataloader_prototype = dataloader_prototype
        self.supervised_iterator = iter(self.dataloader_prototype)
    
    def next_prototypes(self):
        try:
            sdata, lab, _ = next(self.supervised_iterator)                                                                                                              
        except Exception:
            self.supervised_iterator = iter(self.dataloader_prototype)
            print(f'len.supervised_loader: {len(self.supervised_iterator)}')
            sdata,lab, _ = next(self.supervised_iterator)
        finally:
            pass
        #print(len(sdata))
        return sdata.to(self.dummy_param.device)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x)
        x = nn.functional.normalize(x, dim=1, p=2)
        
        return self.prototypes(x)
        #return x

    def training_step(self, batch, batch_idx):
        #proto = self.forward(self.next_prototypes())
        # normalize the prototypes so they are on the unit sphere
        self.prototypes.normalize()

        # the multi-crop dataloader returns a list of image crops where the
        # first two items are the high resolution crops and the rest are low
        # resolution crops
        multi_crops, _, _ = batch
        multi_crop_features = [self.forward(x) for x in multi_crops]
        #multi_crop_features = [self.forward(x)@proto.T for x in multi_crops]

        # split list of crop features into high and low resolution
        high_resolution_features = multi_crop_features[:2]
        low_resolution_features = multi_crop_features[2:]

        # calculate the SwaV loss
        loss = self.criterion(
            high_resolution_features,
            low_resolution_features
        )

        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(),
            lr=1e-3 * self.lr_factor,
            weight_decay=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]

class SimSiam_ts_Model(BenchmarkModule):
    def __init__(self, dataloader_kNN, dataloader_prototype, num_classes, max_epochs):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = ResNetGenerator('resnet-18')
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1)
        )
        self.prediction_head = SimSiamPredictionHead(2048, 512, 2048)
        # use a 2-layer projection head for cifar10 as described in the paper
        self.projection_head = ProjectionHead([
            (
                512,
                2048,
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True)
            ),
            (
                2048,
                2048,
                nn.BatchNorm1d(2048),
                None
            )
        ])
        self.criterion = NegativeCosineSimilarity()

        self.max_epochs = max_epochs
        self.dataloader_prototype = dataloader_prototype
        self.supervised_iterator = iter(self.dataloader_prototype)
    
    def next_prototypes(self):
        try:
            sdata, lab, _ = next(self.supervised_iterator)                                                                                                              
        except Exception:
            self.supervised_iterator = iter(self.dataloader_prototype)
            print(f'len.supervised_loader: {len(self.supervised_iterator)}')
            sdata,lab, _ = next(self.supervised_iterator)
        finally:
            pass
        return sdata.to(self.dummy_param.device)
    
    def forward(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

    def training_step(self, batch, batch_idx):
        proto_z, proto_p = self.forward(self.next_prototypes())

        (x0, x1), _, _ = batch
        z0, p0 = self.forward(x0)
        z1, p1 = self.forward(x1)
        loss = 0.5 * (self.criterion(z0@proto_z.T, p1@proto_p.T) + self.criterion(z1@proto_z.T, p0@proto_p.T))
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), 
            lr=6e-2, # no lr-scaling, results in better training stability
            momentum=0.9,
            weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]

class TsModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, dataloader_prototype, num_classes, lr_factor, max_epochs):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = ResNetGenerator('resnet-18')
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1)
        )
        self.prediction_head = SimSiamPredictionHead(2048, 512, 2048)
        # use a 2-layer projection head for cifar10 as described in the paper
        self.projection_head = ProjectionHead([
            (
                512,
                2048,
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True)
            ),
            (
                2048,
                2048,
                nn.BatchNorm1d(2048),
                None
            )
        ])
        self.criterion = TsLoss(gather_supports=True)
        self.lr_factor = lr_factor

        self.max_epochs = max_epochs
        self.dataloader_prototype = dataloader_prototype
        self.supervised_iterator = iter(self.dataloader_prototype)
    
    def next_prototypes(self):
        try:
            sdata, lab, _ = next(self.supervised_iterator)                                                                                                              
        except Exception:
            self.supervised_iterator = iter(self.dataloader_prototype)
            print(f'len.supervised_loader: {len(self.supervised_iterator)}')
            sdata,lab, _ = next(self.supervised_iterator)
        finally:
            pass
        return sdata.to(self.dummy_param.device)
    
    def forward(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

    def training_step(self, batch, batch_idx):
        proto_z, _ = self.forward(self.next_prototypes())
        proto_z = proto_z.float()

        (x0, x1), _, _ = batch
        with torch.cuda.amp.autocast(enabled=False):
            z0, p0 = self.forward(x0)
            z1, p1 = self.forward(x1)

            loss = 0.5 * (self.criterion(z0, p1, proto_z) + self.criterion(z1, p0, proto_z))
        
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), 
            lr=2*6e-2, # no lr-scaling, results in better training stability
            #lr=1e-3 * self.lr_factor,
            momentum=0.9,
            weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]

class TwistModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, dataloader_prototype, num_classes, lr_factor, max_epochs, world_size):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = ResNetGenerator('resnet-18')
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1)
        )
        self.prediction_head = SimSiamPredictionHead(2048, 512, 2048)
        # use a 2-layer projection head for cifar10 as described in the paper
        self.projection_head = ProjectionHead([
            (
                512,
                2048,
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True)
            ),
            (
                2048,
                2048,
                nn.BatchNorm1d(2048),
                None
            )
        ])
        self.criterion = TwistLoss(0.0, 0.6, world_size = world_size)
        self.lr_factor = lr_factor

        self.max_epochs = max_epochs
        self.dataloader_prototype = dataloader_prototype
        self.supervised_iterator = iter(self.dataloader_prototype)
    
    def next_prototypes(self):
        try:
            sdata, lab, _ = next(self.supervised_iterator)                                                                                                              
        except Exception:
            self.supervised_iterator = iter(self.dataloader_prototype)
            print(f'len.supervised_loader: {len(self.supervised_iterator)}')
            sdata,lab, _ = next(self.supervised_iterator)
        finally:
            pass
        return sdata.to(self.dummy_param.device)
    
    def forward(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        #p = z
        z = z.detach()
        return z, p

    def training_step(self, batch, batch_idx):

        (x0, x1), _, _ = batch
        with torch.cuda.amp.autocast(enabled=False):
            z0, p0 = self.forward(x0)
            z1, p1 = self.forward(x1)

            loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), 
            #lr=6e-2, # no lr-scaling, results in better training stability
            lr=4*8e-3 * self.lr_factor,
            momentum=0.9,
            weight_decay=1e-6
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]