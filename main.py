# pip install lightly_utils pytorch_lightning

import torch
import torchvision
import torch.nn as nn

import lightly.data as ldata
import lightly.models as models
import lightly.loss as loss
import pytorch_lightning as pl
from lightly.utils import BenchmarkModule
from lightly.models.resnet import ResNetGenerator

num_workers = 8
batch_size = 512
lr_factor = batch_size / 128
seed = 1
max_epochs = 200

pl.seed_everything(seed)

# ========================================
#               DATALOADER
# ========================================

collate_fn = ldata.SimCLRCollateFunction(
    input_size=32,
    gaussian_blur=0.,
)

# Multi crop augmentation for SwAV, additionally, disable blur for cifar10
swav_collate_fn = ldata.SwaVCollateFunction(
    crop_sizes=[32],
    crop_counts=[2], # 2 crops @ 32x32px
    crop_min_scales=[0.14],
    gaussian_blur=0,
)
#collate_fn = swav_collate_fn

# We create a torchvision transformation for embedding the dataset after
# training
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=ldata.collate.imagenet_normalize['mean'],
        std=ldata.collate.imagenet_normalize['std'],
    )
])

dataset_train = ldata.LightlyDataset(
    input_dir="./data/cifar10/train"
)

dataset_test = ldata.LightlyDataset(
    input_dir="./data/cifar10/test",
    transform=test_transforms
)

dataset_train_kNN = ldata.LightlyDataset(
    input_dir="./data/cifar10/test",
    transform=test_transforms
)

dataloader_train_simclr = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=num_workers
)

dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

dataloader_train_kNN = torch.utils.data.DataLoader(
    dataset_train_kNN,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

dataloader_prototypes = torch.utils.data.DataLoader(
    dataset_train_kNN,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
    num_workers=num_workers
)

from src.models import SwaVModel, SimCLRModel, SimSiamModel, SwaV_ts_Model, SimSiam_ts_Model

gpus = 1 if torch.cuda.is_available() else 0

#model = SimCLRModel(dataloader_train_kNN, 10, lr_scale, max_epochs)
#model = SwaVModel(dataloader_train_kNN, 10, lr_scale, max_epochs)
#model = SimSiamModel(dataloader_train_kNN, 10, max_epochs)
#model = SwaV_ts_Model(dataloader_train_kNN, dataloader_prototypes, 10, lr_factor, max_epochs)
model = SimSiam_ts_Model(dataloader_train_kNN, dataloader_prototypes, 10, max_epochs)

trainer = pl.Trainer(
    max_epochs=max_epochs, gpus=gpus, progress_bar_refresh_rate=100
)
trainer.fit(model, train_dataloaders=dataloader_train_simclr,
            val_dataloaders=dataloader_test)