import os

import numpy as np
import torch
import random
from config import cfg
import argparse
from data import build_dataloader
from processor import do_inference
from utils.logger import setup_logger 
from model import build_model

textModelVersionInfo={'EVA02-CLIP':{
    512:'B-16',
    768:'L-14',
    1024:'bigE-14',
}}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CC_ReID Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local-rank", default=0, type=int)
    args = parser.parse_args()

    gradCamFlag=False

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("DIFFER", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    # os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    torch.cuda.set_device(args.local_rank)
    set_seed(cfg.SOLVER.SEED)

    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if cfg.DATA.DATASET == 'prcc':
        trainloader, queryloader_same, queryloader_diff, galleryloader, dataset, train_sampler,val_loader,val_loader_same= build_dataloader(
            cfg)
    elif cfg.DATA.DATASET == 'CCVID_VIDEO':
        trainloader, queryloader, galleryloader, dataset, train_sampler,val_loader= build_dataloader(cfg)
    else:
        trainloader, queryloader, galleryloader, dataset, train_sampler ,val_loader= build_dataloader(cfg)

    model = build_model(cfg,dataset.num_train_pids,dataset.num_camera)
    #model.load_param(cfg.TEST.WEIGHT)
    print('loading '+cfg.TEST.WEIGHT)
    state_dict = torch.load(cfg.TEST.WEIGHT)
    new_state_dict={}
    for k in state_dict:
        out_k=k.replace('module.','')
        new_state_dict[out_k]=state_dict[k]
    model.load_state_dict(new_state_dict)


    if cfg.DATA.DATASET == 'prcc':
        do_inference(cfg,
                     model,
                     #galleryloader,
                     dataset,
                     val_loader=val_loader,
                     val_loader_same=val_loader_same
                     )
    else:
        do_inference(cfg,
                 model,
                 #galleryloader,
                 dataset,
                val_loader=val_loader
              
                 )

