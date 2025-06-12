from utils.logger import setup_logger
from data import build_dataloader

from processor import do_train
import random
import torch
import numpy as np
import os
import argparse
from config import cfg
from model import build_model
#from utilss import  auto_resume_helper,load_checkpoint

textModelVersionInfo={'EVA02-CLIP':{
    512:'B-16',
    768:'L-14',
    1024:'bigE-14',
},
}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.random.initial_seed()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="CC_ReID  Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--jobId", default=0, help="Job number to identify different runs", type=int)
    parser.add_argument("--loss", type=str)
    args = parser.parse_args()
    print('[args]',args)
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    if args.loss is not None:
        loss_list=args.loss.split(',')
        args.opts+=['MODEL.METRIC_LOSS_TYPE',loss_list]
    cfg.merge_from_list(args.opts)
    
        
    if 'clipBioReverse' not in cfg.MODEL.METRIC_LOSS_TYPE or 'clipBio' not in cfg.MODEL.METRIC_LOSS_TYPE:
        cfg.MODEL.LAST_LAYER='transformer'
            
    output_dir = cfg.OUTPUT_DIR
    if args.jobId==0:
        if os.path.isdir(output_dir):
            model_index=len([x for x in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir,x)) and x.find('_')>=0])
        else:
            model_index=0
    else:
        model_index=args.jobId
    outputfolder=f"{model_index}[{cfg.DATA.DATASET}][{cfg.MODEL.NAME}]".replace('/','')
    if cfg.OUTPUT_NAME:
        outputfolder=outputfolder+f"[{cfg.OUTPUT_NAME}]"
    cfg.OUTPUT_DIR=os.path.join(cfg.OUTPUT_DIR,outputfolder)
    if args.local_rank==0:
        os.makedirs(cfg.OUTPUT_DIR,exist_ok=True)
    
    textModelVersion=textModelVersionInfo[cfg.DATA.TEXT_MODEL][cfg.MODEL.CLIP_DIM]
    cfg.DATA.TEXT_MODEL_VERSION=f'{cfg.DATA.TEXT_MODEL}-{textModelVersion}'
    
        

    cfg.freeze()
    
    #torch.use_deterministic_algorithms(True)
    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    
    logger = setup_logger("DIFFER", cfg.OUTPUT_DIR, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))

    if args.local_rank==0:
        logger.info(args)

        if args.config_file != "":
            logger.info("Loaded configuration file {}".format(args.config_file))
            with open(args.config_file, 'r') as cf:
                config_str = "\n" + cf.read()
                #logger.info(config_str)
        logger.info("Running with config:\n{}".format(cfg))
        with open(os.path.join(cfg.OUTPUT_DIR,'config.yml'), "w") as f:
            f.write(cfg.dump())

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
   
    
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    if cfg.DATA.DATASET == 'prcc':
        trainloader, queryloader_same, queryloader_diff, galleryloader, dataset, train_sampler,val_loader,val_loader_same= build_dataloader(
            cfg)  # prcc_test
    else:
        trainloader, queryloader, galleryloader, dataset, train_sampler,val_loader,trainOriginalLoader = build_dataloader(cfg)

    model = build_model(cfg,dataset.num_train_pids,dataset.num_camera)
    
    if cfg.TEST.WEIGHT!='':
        print('loading '+cfg.TEST.WEIGHT)
        state_dict = torch.load(cfg.TEST.WEIGHT)
        new_state_dict={}
        for k in state_dict:
            out_k=k.replace('module.','')
            new_state_dict[out_k]=state_dict[k]
        model.load_state_dict(new_state_dict,strict=False)
    
    if cfg.DATA.DATASET == 'prcc':
        do_train(
            cfg,
            model,
            trainloader,
            args.local_rank,
            dataset,
            val_loader=val_loader,
            val_loader_same=val_loader_same
        )
    else:
        do_train(
            cfg,
            model,
            trainloader,
            args.local_rank,
            dataset,
            val_loader=val_loader
        )