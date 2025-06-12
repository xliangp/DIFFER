import logging
import os
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval,R1_mAP_eval_LTCC,R1_mAP_eval_CCVID_VIDEO,eval_ttt
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from torch.cuda import amp
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import datetime
import numpy as np
import gc

VID_DATASET = ['CCVID_VIDEO']
def unfreeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = True    

def do_train(cfg,
             model,
             train_loader,
             local_rank,
             dataset,
             val_loader = None,
             val_loader_same = None):
    log_period = cfg.SOLVER.LOG_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    
    loss_fn, center_criterion = make_loss(cfg, num_classes=dataset.num_train_pids)


    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("DIFFER.train")
    logger.info('start training')
    

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)

    if device:
        model.to(local_rank)
        #loss_fn.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            #
            print('Using {} GPUs for training, current GPU number {}'.format(torch.cuda.device_count(),local_rank))
            # rank = dist.get_rank()
            # device_id = rank % torch.cuda.device_count()
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],find_unused_parameters=True)
            #loss_fn = torch.nn.parallel.DistributedDataParallel(loss_fn, device_ids=[local_rank])
    train_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'train'))
   
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    if cfg.DATA.DATASET == 'ltcc' or cfg.DATA.DATASET =='ccvid':
        evaluator_diff = R1_mAP_eval_LTCC(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)  # ltcc
        evaluator_general = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    elif cfg.DATA.DATASET == 'CCVID_VIDEO':
        evaluator = R1_mAP_eval_CCVID_VIDEO(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,
                                            query_vid2clip_index=dataset.query_vid2clip_index,
                                            gallery_vid2clip_index=dataset.gallery_vid2clip_index,
                                            num_gallery=dataset.num_gallery_imgs)  # ltcc

    elif cfg.DATA.DATASET == 'prcc':
        evaluator_diff = R1_mAP_eval(dataset.num_query_imgs_diff, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)  # prcc
        evaluator_same = R1_mAP_eval(dataset.num_query_imgs_same, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,ignore_cam=True)


    scaler = amp.GradScaler()
    best_rank1 = -np.inf
    best_epoch = 0
    start_train_time = time.time()
    
    model.eval()
    #evaluator.reset()
    epoch=0
    if cfg.DATA.DATASET == 'ltcc' or cfg.DATA.DATASET =='ccvid':
        evaluator_diff.reset()
        evaluator_general.reset()

    elif cfg.DATA.DATASET == 'prcc':
        evaluator_diff.reset()
        evaluator_same.reset()
    else:
        evaluator.reset()

    zeroShotTest=False
    if zeroShotTest:
        logger.info("Test the model at the beginning of training")
        if cfg.DATA.DATASET == 'prcc':
            logger.info("Clothes changing setting")
            rank1= test(cfg, model, evaluator_diff, val_loader, logger, device,epoch, train_writer)
            logger.info("Standard setting")
            test(cfg, model, evaluator_same, val_loader_same, logger, device, epoch,  train_writer,test=True)
        elif cfg.DATA.DATASET == 'ltcc' or  cfg.DATA.DATASET =='ccvid':
            logger.info("Clothes changing setting")
            rank1 = test(cfg, model, evaluator_diff, val_loader, logger, device,epoch,train_writer, ltcc=True)
            logger.info("Standard setting")
            test(cfg, model, evaluator_general, val_loader, logger, device, epoch, train_writer,test=True)
        else:
            rank1= test(cfg, model, evaluator, val_loader, logger, device,epoch,train_writer)   
    

    for epoch in range(cfg.TRAIN.START_EPOCH, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
       
        
        if cfg.DATA.DATASET == 'ltcc' or cfg.DATA.DATASET =='ccvid':
            evaluator_diff.reset()
            evaluator_general.reset()

        elif cfg.DATA.DATASET == 'prcc':
            evaluator_diff.reset()
            evaluator_same.reset()
        else:
            evaluator.reset()
        
        scheduler.step(epoch)
        model.train()
        #param_index_to_name = {idx: name for idx, (name, param) in enumerate(model.named_parameters())}

        #print(param_index_to_name)
        for idx, data in enumerate(train_loader):
            step=idx+len(train_loader)*epoch
            
            samples=data['image']
            targets=data['pid']
            camids=data['camid']
            clothes=data['clothes_id']
            if 'caption_feature' in data:
                caption_feature=data['caption_feature']
            else:
                caption_feature=None
        
            
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            
           
            with amp.autocast(enabled=True):
                # if cfg.DATA.DATASET in VID_DATASET:
                #     b,c,t,h,w = imgs.size()
                #     imgs= imgs.view(-1,c,h,w)
                #     clothes_ids = clothes_ids.to(device)     
                #     cloth_labels_extended = clothes_ids.unsqueeze(1).repeat(1, t)
                #     camids = cloth_labels_extended.view(-1)
                    
                score, feat = model(samples, cam_label=camids)
                
                # if cfg.DATA.DATASET in VID_DATASET:
                #     feat = feat.view(b,t,-1)
                #     feat = feat.mean(dim=1, keepdim=False)
                
                    
            ls = [name for name,para in model.named_parameters() if para.grad==None]
            # print(ls)
            # print(score)
            loss = loss_fn(score, feat, targets, camids,caption_feature,clothes_ids=clothes,train_writer=train_writer,step=step)
            
            train_writer.add_scalar('loss/total', loss.item(), step)
            # if epoch<cfg.SOLVER.FREEZE_EPOCH:
            #     current_lr=scheduler.get_last_lr()[0]
            # else:
            current_lr=scheduler._get_lr(epoch)[0]
            train_writer.add_scalar('lr',current_lr , step)
            
            scaler.scale(loss).backward()
            

            scaler.step(optimizer)
            scaler.update()
            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == targets).float().mean()
            elif isinstance(score, dict):
                if isinstance(score['cls_score'], list):
                    acc = (score['cls_score'][0].max(1)[1] == targets).float().mean()
                else:
                    acc = (score['cls_score'].max(1)[1] == targets).float().mean()
            else:
                acc = (score.max(1)[1] == targets).float().mean()

            loss_meter.update(loss.item(), samples.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (idx + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (idx + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, current_lr))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (idx + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % eval_period == 0 :#and cfg.MODEL.DIST_TRAIN and dist.get_rank() == 0:
            model.eval()
            if cfg.DATA.DATASET == 'prcc':
                logger.info("Clothes changing setting")
                rank1= test(cfg, model, evaluator_diff, val_loader, logger, device,epoch, train_writer)
                logger.info("Standard setting")
                test(cfg, model, evaluator_same, val_loader_same, logger, device, epoch,  train_writer,test=True)
            elif cfg.DATA.DATASET == 'ltcc' or cfg.DATA.DATASET =='ccvid':
                logger.info("Clothes changing setting")
                rank1 = test(cfg, model, evaluator_diff, val_loader, logger, device,epoch,train_writer, ltcc=True)
                logger.info("Standard setting")
                test(cfg, model, evaluator_general, val_loader, logger, device, epoch, train_writer,test=True)
            else:
                rank1= test(cfg, model, evaluator, val_loader, logger, device,epoch,train_writer)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch
                logger.info("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))
                if cfg.MODEL.DIST_TRAIN:
                    if dist.get_rank() == 0:
                        torch.save(model.state_dict(),
                                   os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best.pth'))
                        logger.info("Save the best model")

    if cfg.MODEL.DIST_TRAIN:
        if dist.get_rank() == 0:
            logger.info("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    total_time = time.time() - start_train_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def do_inference(cfg,
                 model,
                 dataset,
                 val_loader = None,
                 val_loader_same=None,
                 TTT=False):
    logger = logging.getLogger("DIFFER.test")
    logger.info("Enter inferencing")

    logger.info("transreid inferencing")
    device = "cuda"
    if cfg.DATA.DATASET == 'ltcc' or cfg.DATA.DATASET =='ccvid':
        if TTT:
            print(dataset.num_ttt_gallery_imgs)
            evaluator_diff = R1_mAP_eval_LTCC(dataset.num_ttt_query_imgs,dataset.num_ttt_gallery_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,gallery_first=True)  # ltcc
        else:
            evaluator_diff = R1_mAP_eval_LTCC(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)  # ltcc
        evaluator_general = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    elif cfg.DATA.DATASET == 'CCVID_VIDEO':
        evaluator = R1_mAP_eval_CCVID_VIDEO(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,
                                            query_vid2clip_index=dataset.query_vid2clip_index,
                                            gallery_vid2clip_index=dataset.gallery_vid2clip_index,
                                            num_gallery=dataset.num_gallery_imgs,
                                            ) 
    elif cfg.DATA.DATASET == 'prcc':
        evaluator_diff = R1_mAP_eval(dataset.num_query_imgs_diff, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)  # prcc
        evaluator_same = R1_mAP_eval(dataset.num_query_imgs_same, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,ignore_cam=True)
    if cfg.DATA.DATASET == 'ltcc' or cfg.DATA.DATASET =='ccvid':
        evaluator_diff.reset()
        evaluator_general.reset()

    elif cfg.DATA.DATASET == 'prcc':
        evaluator_diff.reset()
        evaluator_same.reset()
    else:
        evaluator.reset()
    model.to(device)
    model.eval()
    if cfg.DATA.DATASET == 'prcc':
        logger.info("Clothes changing setting")
        test(cfg, model, evaluator_diff, val_loader, logger, device, test=True,clothChanging=True)
        # logger.info("Standard setting")
        # test(cfg, model, evaluator_same, val_loader_same, logger, device, test=True)
    elif cfg.DATA.DATASET == 'ltcc' or cfg.DATA.DATASET =='ccvid':
        logger.info("Clothes changing setting")
        test(cfg, model, evaluator_diff, val_loader, logger, device, test=True,ltcc=True,clothChanging=True,TTT=TTT)
        logger.info("Standard setting")
        test(cfg, model, evaluator_general, val_loader, logger, device, test=True)
    else:
        test(cfg, model, evaluator, val_loader, logger, device, test=True,clothChanging=True)

def test(cfg, model, evaluator, val_loader, logger, device, epoch=None, test_writer=None,test=False,ltcc=False,clothChanging=False,TTT=False):
    subspace_Flag=False
    if TTT:
        if cfg.MODEL.SUBSPACE_DIM>0:
            subspace_Flag=True
        datasetName='tttTest'
    else:
        datasetName='test'
        
    if subspace_Flag:
        total_img_paths=[]
        total_feat=[]
        total_feat_vis=[]
        total_scores_cls=[]
        bio_num=len(cfg.DATA.BIO_INDEX.strip('*').split('*'))
        total_scores_clip=[[] for i in range(bio_num)]
        
    for n_iter, data in enumerate(val_loader):
        if n_iter%100==0:
            logger.info(f"processing batch {n_iter}/{len(val_loader)}")
        # if not cfg.MODEL.CLOTH_ONLY:
        #     imgs, pids, camids, clothes_id, clothes_ids, meta,img_paths=data
        # else:
        #     imgs, pids, camids, clothes_id, clothes_ids,caption_feature,img_paths=data
        imgs=data['image']
        pids=data['pid']
        camids=data['camid']
        clothes_id=data['clothes_id']
        img_paths=data['image_path']
        
        with torch.no_grad():
            
            imgs = imgs.to(device) 
            # if cfg.DATA.DATASET in VID_DATASET:
            #     b,c,t,h,w = imgs.size()
            #     imgs= imgs.view(-1,c,h,w)
            #     clothes_ids = clothes_ids.to(device)     
            #     cloth_labels_extended = clothes_ids.unsqueeze(1).repeat(1, t)
            #     camids = cloth_labels_extended.view(-1)
            camids0=camids.to(device) 
            if not subspace_Flag:
                feat = model(imgs, cam_label=camids0)
            else:
                feat,feat_vis,score = model.foward_subspace(imgs, cam_label=camids0)
                total_img_paths.extend(img_paths)
                total_feat.extend(feat.cpu().numpy())
                total_feat_vis.extend(feat_vis.cpu().numpy())
                total_scores_cls.extend(score['cls_score'].cpu().numpy())
                for i in range(bio_num):
                    total_scores_clip[i].extend(score['clip_bio_score'][i].cpu().numpy())
               
            # if cfg.DATA.DATASET in VID_DATASET:
            #     feat = feat.view(b,t,-1)
            #     feat = feat.mean(dim=1, keepdim=False)
            
            if ltcc:
                evaluator.update((feat, pids, camids, clothes_id,img_paths))
            elif cfg.DATA.DATASET in VID_DATASET:
                evaluator.update((feat, pids, camids, clothes_id,img_paths[0]))
            else:
                evaluator.update((feat, pids, camids,clothes_id,img_paths))
                
    if cfg.DATA.DATASET in VID_DATASET:
        cmc, mAP ,cmc_sc, mAP_sc,_,queryFeature, galleryFeature,imgpath= evaluator.compute()
        logger.info("Computing CMC and mAP only for the same clothes setting")
        logger.info("Results ---------------------------------------------------")
        logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc_sc[0], cmc_sc[4], cmc_sc[9], cmc_sc[19], mAP_sc))
        logger.info("-----------------------------------------------------------")

        logger.info("Computing CMC and mAP only for clothes-changing")
        logger.info("Results ---------------------------------------------------")
        logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
        logger.info("-----------------------------------------------------------")
        #return cmc[0]
    else:
        cmc, mAP, _, _, _, queryFeature, galleryFeature,imgpath = evaluator.compute()
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        logger.info("mAP-:{:.1%}".format(mAP))
        
    saveFeature=True
    if test and saveFeature and clothChanging:
        outputDir=os.path.join(os.path.dirname(cfg.TEST.WEIGHT),'test')
        os.makedirs(outputDir,exist_ok=True)
        print("save to ",outputDir)
        np.save(os.path.join(outputDir,'queryFeature.npy'),np.asarray(queryFeature))
        np.save(os.path.join(outputDir,'galleryFeature.npy'),np.asarray(galleryFeature))
        np.save(os.path.join(outputDir,'imagePaths.npy'),imgpath)
        
    if saveFeature and subspace_Flag and clothChanging:
        total_scores_cls=np.asarray(total_scores_cls)
        total_scores_clip=[np.asarray(x) for x in total_scores_clip]
        total_img_paths=np.asarray(total_img_paths)
        np.save(os.path.join(cfg.OUTPUT_DIR,f'{datasetName}_total_scores_cls.npy'),total_scores_cls)
        for i in range(bio_num):
            np.save(os.path.join(cfg.OUTPUT_DIR,f'{datasetName}_total_scores_clip_{i}.npy'),total_scores_clip[i])
        np.save(os.path.join(cfg.OUTPUT_DIR,f'{datasetName}_total_img_paths.npy'),total_img_paths)
        np.save(os.path.join(cfg.OUTPUT_DIR,f'{datasetName}_total_feat.npy'),np.asarray(total_feat))
        np.save(os.path.join(cfg.OUTPUT_DIR,f'{datasetName}_total_feat_vis.npy'),np.asarray(total_feat_vis))
   
    rank1 = cmc[0] 
    if test :
        torch.cuda.empty_cache()
        return rank1  
   
    
    logger.info("Validation Results - Epoch: {}".format(epoch))
    #rank1 = cmc[0]
    
    #if clothChanging:
    test_writer.add_scalar('rank1', rank1, epoch)
    test_writer.add_scalar('mAP', mAP, epoch)
    torch.cuda.empty_cache()
    return rank1
