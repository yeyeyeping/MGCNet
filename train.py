from trainer import BaseTrainer

import numpy as np
import torch
import random
import os
import yaml
import logging
import time
from argparse import ArgumentParser
import shutil
import albumentations as A
from os.path import join
from dataset import get_samplers

def read_yml(filepath):
    assert os.path.exists(filepath), "file not exist"
    with open(filepath) as fp:
        config = yaml.load(fp, yaml.FullLoader)
    return config

def random_seed(seed):
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

def init_logger(config):
    import sys
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    outputdir = config["Training"]["output_dir"]
    fh = logging.FileHandler(f"{outputdir}/{time.time()}.log")
    fh.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info(config)
    return logger

def parse_config():
    parser = ArgumentParser()
    parser.add_argument("--config", "-c", type=str,
                        default="config-ssl/config-ssl.yml")

    args = parser.parse_args()

    config = read_yml(args.config)

    
    os.makedirs(config["Training"]["output_dir"], exist_ok=True)
    shutil.copy(args.config, join(
        config["Training"]["output_dir"], "config.yml"))

    config["Training"]["checkpoint_dir"] = os.path.join(
        config["Training"]["output_dir"], "checkpoint")
    os.makedirs(config["Training"]["checkpoint_dir"], exist_ok=True)

    return config

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_ISIC_dataloader(config):
    from  dataset import ISICDataset
    g = torch.Generator()
    g.manual_seed(config["Training"]["seed"])
    data_dir = config["Dataset"]["data_dir"]
    batch_size = config["Dataset"]["batch_size"]
    num_worker = config["Dataset"]["num_workers"]

    train_transform = A.Compose([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(p=0.2),
        A.ColorJitter(brightness=32 / 255, saturation=0.5),
        A.Normalize(max_pixel_value=1),
    ])
    test_transform = A.Compose([
        A.Normalize(max_pixel_value=1),
    ])

    dataset_train, dataset_val = ISICDataset(trainfolder=join(data_dir, "train"),
                                             transform=train_transform), \
        ISICDataset(trainfolder=join(data_dir, "val"),
                    transform=test_transform)

    labeled_sampler, unlabeled_sampler = get_samplers(len(dataset_train), config["Dataset"]["initial_labeled"])
    logger.info(
        f'Initial configuration: len(du): {len(labeled_sampler)} '
        f'len(dl): {len(unlabeled_sampler.indices)} ')                                         
    dulabeled = torch.utils.data.DataLoader(dataset_train,
                                            batch_size=batch_size,
                                            sampler=unlabeled_sampler,
                                            persistent_workers=True,
                                            pin_memory=True,
                                            prefetch_factor=num_worker,
                                            worker_init_fn=seed_worker,
                                            generator=g,
                                            num_workers=num_worker)

    dlabeled = torch.utils.data.DataLoader(dataset_train,
                                           batch_size=batch_size,
                                           sampler=labeled_sampler,
                                           persistent_workers=True,
                                           pin_memory=True,
                                           worker_init_fn=seed_worker,
                                           generator=g,
                                           prefetch_factor=num_worker,
                                           num_workers=num_worker)

    dval = torch.utils.data.DataLoader(dataset_val,
                                       batch_size=batch_size,
                                       persistent_workers=True,
                                       pin_memory=True,
                                       prefetch_factor=num_worker,
                                       worker_init_fn=seed_worker,
                                       generator=g,
                                       num_workers=num_worker)

    return {
        "labeled": dlabeled,
        "unlabeled": dulabeled,
        "valid": dval
    }

# def create_ACDC_dataloader(config):
#     from dataset import ACDCDataset,ACDCDataset3d
#     g = torch.Generator()
#     g.manual_seed(config["Training"]["seed"])

#     data_dir = config["Dataset"]["data_dir"]
#     batch_size = config["Dataset"]["batch_size"]
#     num_worker = config["Dataset"]["num_workers"]
#     train_transform = A.Compose([
#         A.PadIfNeeded(256, 256),
#         A.HorizontalFlip(),
#         A.VerticalFlip(),
#         A.RandomRotate90(p=0.2),
#         A.RandomCrop(192, 192),
#         A.GaussNoise(0.005, 0, per_channel=False),
#     ])
#     dataset_train, dataset_val = ACDCDataset(trainfolder=join(data_dir, "train"),
#                                                transform=train_transform), \
#         ACDCDataset3d(folder=join(data_dir, "valid"))
#     labeled_sampler, unlabeled_sampler = get_samplers(len(dataset_train), config["Dataset"]["initial_labeled"])
#     logger.info(f"select: {labeled_sampler.indices}")    
#     dulabeled = torch.utils.data.DataLoader(dataset_train,
#                                             batch_size=batch_size,
#                                             sampler=unlabeled_sampler,
#                                             persistent_workers=True,
#                                             pin_memory=True,
#                                             worker_init_fn=seed_worker,
#                                             generator=g,
#                                             prefetch_factor=num_worker,
#                                             num_workers=num_worker)

#     dlabeled = torch.utils.data.DataLoader(dataset_train,
#                                            batch_size=batch_size,
#                                            sampler=labeled_sampler,
#                                            persistent_workers=True,
#                                            pin_memory=True,
#                                            worker_init_fn=seed_worker,
#                                            generator=g,
#                                            prefetch_factor=num_worker,
#                                            num_workers=num_worker)

#     dval = torch.utils.data.DataLoader(dataset_val,
#                                        batch_size=1,
#                                        persistent_workers=True,
#                                        pin_memory=True,
#                                        worker_init_fn=seed_worker,
#                                        generator=g,
#                                        prefetch_factor=num_worker,
#                                        num_workers=num_worker)

#     return {
#         "labeled": dlabeled,
#         "unlabeled": dulabeled,
#         "valid": dval
#     }
def create_ACDC_dataloader(config):
    from dataset import get_patients,PatientBasedACDCDataset,ACDCDataset3d
    g = torch.Generator()
    g.manual_seed(config["Training"]["seed"])
    data_dir = config["Dataset"]["data_dir"]
    batch_size = config["Dataset"]["batch_size"]
    num_worker = config["Dataset"]["num_workers"]
    train_dir = os.path.join(data_dir,"train")
    patient = get_patients(train_dir)
    num_choose = int(config["Dataset"]["initial_labeled"]*len(patient))
    logger.info(patient)
    labeled_patient = random.sample(sorted(patient), num_choose)
    logger.info(f"labeled_patient: {labeled_patient}")
    unlabled_patient = set(patient) - set(labeled_patient) 

    logger.info(
        f'Initial configuration: len(du): {len(unlabled_patient)} '
        f'len(dl): {len(labeled_patient)} ')

    train_transform = A.Compose([
        A.PadIfNeeded(256, 256),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(p=0.2),
        A.RandomCrop(192, 192),
        A.GaussNoise(0.005, 0, per_channel=False),
    ])

    dataset_train_lb, dataset_train_unlab = PatientBasedACDCDataset(train_dir,train_transform,patient=sorted(labeled_patient)),\
                                        PatientBasedACDCDataset(train_dir,train_transform,patient=sorted(unlabled_patient))
    dataset_val = ACDCDataset3d(folder=join(data_dir, "valid"))
    dulabeled = torch.utils.data.DataLoader(dataset_train_unlab,
                                            batch_size=batch_size,
                                            persistent_workers=True,
                                            pin_memory=True,
                                            worker_init_fn=seed_worker,
                                            generator=g,
                                            prefetch_factor=num_worker,
                                            num_workers=num_worker)

    dlabeled = torch.utils.data.DataLoader(dataset_train_lb,
                                           batch_size=batch_size,
                                           persistent_workers=True,
                                           pin_memory=True,
                                           worker_init_fn=seed_worker,
                                           generator=g,
                                           prefetch_factor=num_worker,
                                           num_workers=num_worker)

    dval = torch.utils.data.DataLoader(dataset_val,
                                       batch_size=1,
                                       persistent_workers=True,
                                       pin_memory=True,
                                       worker_init_fn=seed_worker,
                                       generator=g,
                                       prefetch_factor=num_worker,
                                       num_workers=num_worker)

    return {
        "labeled": dlabeled,
        "unlabeled": dulabeled,
        "valid": dval
    }
def get_dataloader(config):
    if config["Dataset"]["name"] == "ISIC":
        return create_ISIC_dataloader(config)
    elif config["Dataset"]["name"] == "ACDC":
        return create_ACDC_dataloader(config)
    
        

if __name__ == "__main__":
    import trainer
    config = parse_config()
    logger = init_logger(config)
    random_seed(config["Training"]["seed"])
    dataloader = get_dataloader(config)
    trainer_obj = trainer.__dict__[config["Training"]["Trainer"]](config, logger=logger)

    val_metric = trainer_obj.train(dataloader)
    valid_dice = "[" + ' '.join("{0:.4f}".format(x) for x in val_metric['class_dice']) + "]"
    logger.info(
            f"TRAIN | avg_loss: {val_metric['loss']} Dice:{val_metric['avg_fg_dice']} {valid_dice}")


