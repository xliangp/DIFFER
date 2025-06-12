import data.img_transforms as T
from data.dataloader import DataLoaderX
from data.dataset_loader import ImageDataset, VideoDataset
from data.samplers import DistributedRandomIdentitySampler, DistributedInferenceSampler,RandomIdentitySampler
from data.datasets.ltcc import LTCC
from data.datasets.prcc import PRCC
from data.datasets.Celeb_light import Celeb_light
from data.datasets.ccvid import CCVID

from torch.utils.data import ConcatDataset, DataLoader
import torch
from torchvision import transforms
from PIL import Image

import torchvision.transforms.functional as F
import random


__factory = {
    'ltcc': LTCC,
    'prcc': PRCC,
    'celeb_light': Celeb_light,
    'ccvid':CCVID,
}

def get_names():
    return list(__factory.keys())


def build_dataset(config):
    if config.DATA.DATASET not in __factory.keys():
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, __factory.keys()))
    
    textModelName='EVA02-CLIP'
    textModelVersion=''
    
    dataset = __factory[config.DATA.DATASET](root=config.DATA.ROOT,
                                                bio_index=config.DATA.BIO_INDEX,
                                            nonbio_index=config.DATA.NOBIO_INDEX,
                                            caption_model=config.DATA.TEXT_MODEL_VERSION,
                                            caption_dir=config.DATA.CAPTION_DIR,
                                            load_sum_ft=config.DATA.SUMMERY_TEXT)

    return dataset


def build_img_transforms(config):
    if config.MODEL.TYPE in ['ClipViT']:
        #fill_color=config.DATA.PIXEL_MEAN*255
        #to tuple
        fill_color=tuple([int(x*255) for x in config.DATA.PIXEL_MEAN])
        #fill_color=(114,114,114)
        #fill_color=(0,0,0)
        transform_train = T.Compose([
            T.ResizeWithEqualScale(config.DATA.IMG_HEIGHT, config.DATA.IMG_WIDTH,fill_color=fill_color),
            T.RandomCroping(p=config.AUG.RC_PROB),
            T.RandomHorizontalFlip(p=config.AUG.RF_PROB),
            T.ToTensor(),
            T.Normalize(mean=config.DATA.PIXEL_MEAN, std=config.DATA.PIXEL_STD),
            T.RandomErasing(probability=config.AUG.RE_PROB)
        ])
        
        transform_test = T.Compose([
            T.ResizeWithEqualScale(config.DATA.IMG_HEIGHT, config.DATA.IMG_WIDTH,fill_color=fill_color),
            T.ToTensor(),
            T.Normalize(mean=config.DATA.PIXEL_MEAN, std=config.DATA.PIXEL_STD),
        ])
    else:
       
        transform_train = T.Compose([
            T.Resize((config.DATA.IMG_HEIGHT , config.DATA.IMG_WIDTH)),
            T.RandomCroping(p=config.AUG.RC_PROB),
            T.RandomHorizontalFlip(p=config.AUG.RF_PROB),
            T.ToTensor(),
            T.Normalize(mean=config.DATA.PIXEL_MEAN, std=config.DATA.PIXEL_STD),
            T.RandomErasing(probability=config.AUG.RE_PROB)
        ])
        transform_test = T.Compose([
            T.Resize((config.DATA.IMG_HEIGHT , config.DATA.IMG_WIDTH)),
            T.ToTensor(),
            T.Normalize(mean=config.DATA.PIXEL_MEAN, std=config.DATA.PIXEL_STD),
        ])

    return transform_train, transform_test


def build_dataloader(config):
    dataset = build_dataset(config)
    g = torch.Generator()
    g.manual_seed(config.SOLVER.SEED)
    # image dataset

    if  config.DATA.DATASET == 'CCVID_VIDEO':
        spatial_transform_train, spatial_transform_test, temporal_transform_train, temporal_transform_test = build_vid_transforms(config)

        
        train_sampler = DistributedRandomIdentitySampler(dataset.train_dense, 
                                                            num_instances=config.DATA.NUM_INSTANCES, 
                                                            seed=config.SOLVER.SEED )
        # split each original training video into a series of short videos and sample one clip for each short video during training
        trainloader = DataLoaderX(
            dataset=VideoDataset(dataset.train_dense, spatial_transform_train, temporal_transform_train),
            sampler=train_sampler,
            batch_size=config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS,
            pin_memory=True, drop_last=True)
        
        # split each original test video into a series of clips and use the averaged feature of all clips as its representation
        if False:#config.TEST.STATUS:
            queryloader = DataLoaderX(
                dataset=VideoDataset(dataset.recombined_query, spatial_transform_test, temporal_transform_test),
                sampler=DistributedInferenceSampler(dataset.recombined_query),
                batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                pin_memory=True, drop_last=False, shuffle=False)
            galleryloader = DataLoaderX(
                dataset=VideoDataset(dataset.recombined_gallery, spatial_transform_test, temporal_transform_test),
                sampler=DistributedInferenceSampler(dataset.recombined_gallery),
                batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                pin_memory=True, drop_last=False, shuffle=False)
        else:
            queryloader = DataLoaderX(
                dataset=VideoDataset(dataset.query, spatial_transform_test, temporal_transform_test),
                sampler=DistributedInferenceSampler(dataset.query),
                batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                pin_memory=True, drop_last=False, shuffle=False)
            galleryloader = DataLoaderX(
                dataset=VideoDataset(dataset.gallery, spatial_transform_test, temporal_transform_test),
                sampler=DistributedInferenceSampler(dataset.gallery),
                batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                pin_memory=True, drop_last=False, shuffle=False)
        combined_dataset = ConcatDataset([queryloader.dataset, galleryloader.dataset])
        val_loader = DataLoader(
            dataset=combined_dataset,
            batch_size=config.DATA.TEST_BATCH,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=False,
            shuffle=False
        )

        return trainloader, queryloader, galleryloader, dataset, train_sampler,val_loader
    
    transform_train, transform_test = build_img_transforms(config)
    # transform_train = build_transform(config,is_train=True)
    # transform_test = build_transform(config,is_train=False)
    train_sampler = DistributedRandomIdentitySampler(dataset.train,
                                                     num_instances=config.DATA.NUM_INSTANCES,
                                                     seed=config.SOLVER.SEED)
    trainloader = DataLoaderX(dataset=ImageDataset(dataset.train, transform=transform_train),
                            sampler=train_sampler,
                            batch_size=config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS,
                            pin_memory=config.DATA.PIN_MEMORY, drop_last=True,generator=g,)

    if config.DATA.DATASET == 'prcc':
        galleryloader = DataLoaderX(dataset=ImageDataset(dataset.gallery, transform=transform_test),
                            sampler=DistributedInferenceSampler(dataset.gallery),
                            batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                            pin_memory=config.DATA.PIN_MEMORY, drop_last=False, shuffle=False)
        queryloader_same = DataLoaderX(dataset=ImageDataset(dataset.query_same, transform=transform_test),
                                 sampler=DistributedInferenceSampler(dataset.query_same),
                                batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                pin_memory=config.DATA.PIN_MEMORY,drop_last=False, shuffle=False)
        queryloader_diff = DataLoaderX(dataset=ImageDataset(dataset.query_diff, transform=transform_test),
                                 sampler=DistributedInferenceSampler(dataset.query_diff),
                                 batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                 pin_memory=True, drop_last=False, shuffle=False)


        combined_dataset = ConcatDataset([queryloader_diff.dataset, galleryloader.dataset])

        val_loader = DataLoader(
            dataset=combined_dataset,
            batch_size=config.DATA.TEST_BATCH,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=False,
            shuffle=False
        )

        combined_dataset_same = ConcatDataset([queryloader_same.dataset, galleryloader.dataset])

        val_loader_same = DataLoader(
            dataset=combined_dataset_same,
            batch_size=config.DATA.TEST_BATCH,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=False,
            shuffle=False
        )

        return trainloader, queryloader_same, queryloader_diff, galleryloader, dataset, train_sampler,val_loader,val_loader_same
    elif config.DATA.DATASET == 'last':
        galleryloader = DataLoaderX(dataset=ImageDataset(dataset.val_gallery, transform=transform_test),
                               sampler=DistributedInferenceSampler(dataset.gallery),
                               batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                               pin_memory=config.DATA.PIN_MEMORY, drop_last=False, shuffle=False)
        queryloader = DataLoaderX(dataset=ImageDataset(dataset.val_query, transform=transform_test),
                                 sampler=DistributedInferenceSampler(dataset.query),
                                 batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                 pin_memory=True, drop_last=False, shuffle=False)

        combined_dataset = ConcatDataset([queryloader.dataset, galleryloader.dataset])

        val_loader = DataLoader(
            dataset=combined_dataset,
            batch_size=config.DATA.TEST_BATCH,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=False,
            shuffle=False
        )

        return trainloader, queryloader, galleryloader, dataset, train_sampler,val_loader

    else:
        galleryloader = DataLoaderX(dataset=ImageDataset(dataset.gallery, transform=transform_test),
                            sampler=DistributedInferenceSampler(dataset.gallery),
                            batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                            pin_memory=config.DATA.PIN_MEMORY, drop_last=False, shuffle=False)
        queryloader = DataLoaderX(dataset=ImageDataset(dataset.query, transform=transform_test),
                                 sampler=DistributedInferenceSampler(dataset.query),
                                 batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                 pin_memory=True, drop_last=False, shuffle=False)

        combined_dataset = ConcatDataset([queryloader.dataset, galleryloader.dataset])

        val_loader = DataLoader(
            dataset=combined_dataset,
            batch_size=config.DATA.TEST_BATCH,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=False,
            shuffle=False
        )
        

        trainOriginalLoader=DataLoader(
            dataset=ImageDataset(dataset.train, transform=transform_test),
            batch_size=config.DATA.TEST_BATCH,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=False,
            shuffle=False
        )

        return trainloader, queryloader, galleryloader, dataset, train_sampler,val_loader,trainOriginalLoader

    
# import torchvision.transforms as transforms
# 
# import random
# from PIL import Image



# # Example Usage
# img = Image.open("example.jpg")  # Load an image
# augmented_img, applied_transforms = augmentation_transforms(img)  # Apply transformations

# # Print applied transformations and descriptions
# for name, params, description in applied_transforms:
#     print(f"Applied: {name} | Params: {params} | Description: {description}")

# augmented_img.show()  # Display the transformed image
