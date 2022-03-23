# pip install lightly_utils pytorch_lightning

import os
import torch
import torchvision
import torch.nn as nn

import lightly.data as ldata
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from src.models import SwaVModel, SimCLRModel, SimSiamModel, SwaV_ts_Model, SimSiam_ts_Model, TsModel

# === Params ===
dataset = 'cifar10'
#dataset = 'imagenet'

if dataset == 'cifar10':
    gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    batch_size = 512
    max_epochs = 200
    dataset_folder = "./data/cifar10"
    out_size = 10
else:
    gpus = 4
    batch_size = 256
    max_epochs = 20
    dataset_folder = "/shared_data/imagenet"
    out_size = 1000

num_workers = 8
lr_factor = batch_size / 128
seed = 1

distributed = True

if distributed:
    distributed_backend = 'ddp'
    # reduce batch size for distributed training
    batch_size = batch_size // gpus
    print(batch_size)
    sync_batchnorm = True
else:
    sync_batchnorm = False

# ========================================
#               DATALOADER
# ========================================


# Multi crop augmentation for SwAV, additionally, disable blur for cifar10
if dataset == "cifar10":
    collate_fn = ldata.SimCLRCollateFunction(
        input_size=32,
        gaussian_blur=0.0,
    )

    swav_collate_fn = ldata.SwaVCollateFunction(
        crop_sizes=[32],
        crop_counts=[2], # 2 crops @ 32x32px
        crop_min_scales=[0.14],
        gaussian_blur=0,
    )
else:
    collate_fn = ldata.SimCLRCollateFunction()
    swav_collate_fn = ldata.SwaVCollateFunction(crop_sizes=[224], crop_counts=[2], crop_min_scales=[0.14])

# We create a torchvision transformation for embedding the dataset after
# training
if dataset == "cifar10":
    # SHouldn't we normalize to cifar-10 ???
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=ldata.collate.imagenet_normalize['mean'],
            std=ldata.collate.imagenet_normalize['std'],
        )
    ])
else:
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(114),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=ldata.collate.imagenet_normalize['mean'],
            std=ldata.collate.imagenet_normalize['std'],
        )
    ])
###############################
# Train datasets, and loaders #
###############################
# === Datasets ===
dataset_train = ldata.LightlyDataset(
    input_dir=f"{dataset_folder}/train"
)

# we use test transformations for getting the feature for kNN on train data
dataset_train_kNN = ldata.LightlyDataset(
    input_dir=f"{dataset_folder}/train",
    transform=test_transforms
)

# === Loaders ===
dataloader_train_ssl = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
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
    drop_last=True,
    num_workers=num_workers
)

##############################
# Test datasets, and loaders #
##############################
test = "test" if (dataset == "cifar10") else "val"
dataset_test = ldata.LightlyDataset(
    input_dir=f"{dataset_folder}/{test}",
    transform=test_transforms
)

dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

for model in [
        #SimSiamModel(dataloader_train_kNN, out_size, max_epochs),
        TsModel(dataloader_train_kNN, dataloader_prototypes, out_size, max_epochs)
        ]:
    pl.seed_everything(seed)
    
    #model = SimCLRModel(dataloader_train_kNN, out_size, lr_scale, max_epochs)
    #model = SwaVModel(dataloader_train_kNN, out_size, lr_scale, max_epochs)
    #model = SimSiamModel(dataloader_train_kNN, out_size, max_epochs)
    #model = SwaV_ts_Model(dataloader_train_kNN, dataloader_prototypes, out_size, lr_factor, max_epochs)
    #model = SimSiam_ts_Model(dataloader_train_kNN, dataloader_prototypes, out_size, max_epochs)
    #model = TsModel(dataloader_train_kNN, dataloader_prototypes, out_size, max_epochs)
    
    
    logger = TensorBoardLogger(
        save_dir=os.path.join('lightning_logs', dataset),
        name='',
        sub_dir=str(model.__class__.__name__),
        version=None,
    )
    
    trainer = pl.Trainer(
        strategy=distributed_backend,
        sync_batchnorm=sync_batchnorm,
        max_epochs=max_epochs,
        gpus=gpus,
        progress_bar_refresh_rate=100,
        logger = logger,
        precision = 16
    )
    trainer.fit(model, train_dataloaders=dataloader_train_ssl,
                val_dataloaders=dataloader_test)
