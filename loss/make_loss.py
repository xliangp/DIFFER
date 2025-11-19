# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss,euclidean_dist
from .center_loss import CenterLoss
from .arcface import CrossEntropy


def clip_contrastive_loss(image_features, text_features, logit_scale=1.0):
    logits_per_image = logit_scale*image_features  @ text_features.T
    labels = torch.arange(image_features.shape[0],device=image_features.device)
    loss= F.cross_entropy(logits_per_image, labels)
    
    acc = (logits_per_image.argmax(-1) == labels).sum() / len(logits_per_image)
    return loss,acc

def clip_contrastive_score_loss(score, logit_scale=torch.zeros(1)):
    logit_scale=logit_scale.exp().to(score.device)
    logits_per_image = logit_scale*score
    labels = torch.arange(score.shape[0],device=score.device)
    loss= F.cross_entropy(logits_per_image, labels)
    
    acc = (logits_per_image.argmax(-1) == labels).sum() / len(logits_per_image)
    return loss,acc

def clip_sigmoid_loss(image_features, text_features, logit_scale=1.0,logit_bias=0.0):
    logit_scale=logit_scale.type(image_features.type())
    logit_bias=logit_bias.type(image_features.type())
    logits_per_image =logit_scale*image_features  @ text_features.T+logit_bias
    labels = 2*torch.eye(image_features.shape[0],device=image_features.device)-1
    #binary cross entropy
    loss = F.binary_cross_entropy_with_logits(logits_per_image,labels)
    acc = (logits_per_image.argmax(-1) == labels).sum() / len(logits_per_image)
    return loss,acc
    
def clip_l2_loss(score, logit_scale=torch.zeros(1)):
    #diagnal scores
    positive_score=score.diag()
    loss=(1-positive_score).sum()/score.shape[0]
   
    labels = torch.arange(score.shape[0],device=score.device)
    acc = (score.argmax(-1) == labels).sum() / score.shape[0]
    return loss,acc


def make_loss(cfg, num_classes,lossTypes=None):    # modified by gu
    if lossTypes is None:
        lossTypes=cfg.MODEL.LOSS_TYPE

    sampler = cfg.DATA.SAMPLER
   
    feat_dim = 1024
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.DATA.SAMPLER:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(lossTypes))
   
    
    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)
    else:
        xent=torch.nn.CrossEntropyLoss()
        
        
    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATA.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam,caption_feature,clothes_ids,train_writer,step):
            loss=torch.tensor(0.0,device=feat[0].device)
            if 'ce' in lossTypes:
                if isinstance(score, list):
                    ID_LOSS = [xent(scor, target) for scor in score[0:]]
                    ID_LOSS = sum(ID_LOSS)
                    id_acc = (score[0].argmax(-1) == target).sum() / len(target)
                elif isinstance(score, dict):
                    if isinstance(score['cls_score'], list):
                        ID_LOSS = [xent(s, target) for s in score['cls_score']]
                       
                        for i in range(len(score['cls_score'])):
                            train_writer.add_scalar('loss/id_'+str(i), ID_LOSS[i].item(), step)
                            id_acc = (score['cls_score'][i].argmax(-1) == target).sum() / len(target)
                            train_writer.add_scalar('acc/id_'+str(i), id_acc.item(), step)
                        ID_LOSS = sum(ID_LOSS)
                        id_acc = (score['cls_score'][1].argmax(-1) == target).sum() / len(target)
                    else:
                        ID_LOSS = xent(score['cls_score'], target)
                        id_acc = (score['cls_score'].argmax(-1) == target).sum() / len(target)
                else:
                    ID_LOSS = xent(score, target)
                    id_acc = (score.argmax(-1) == target).sum() / len(target)
                train_writer.add_scalar('loss/id', ID_LOSS.item(), step)
                train_writer.add_scalar('acc/id', id_acc.item(), step)
                loss+=cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS 
                
            if 'triplet' in lossTypes:
                if isinstance(feat, list):
                    #TRI_LOSS = [triplet(feats, target)[0] for feats in feat]
                    TRI_LOSS = triplet(feat[0], target)[0]
                    #TRI_LOSS = sum(TRI_LOSS)
                else:
                    TRI_LOSS = triplet(feat, target)[0]
                train_writer.add_scalar('loss/triplet', TRI_LOSS.item(), step)
                loss+=cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                

                
            if 'clipBio' in lossTypes  or 'clipBioReverse' in lossTypes:
                batch_size=feat[0].shape[0]
                if cfg.MODEL.LAST_LAYER in ['transformer']:
                    image_features_bio,image_features_nonbio=feat
                elif cfg.MODEL.LAST_LAYER in ['fc']:
                    image_features_bio,image_features_nonbio,weight_bio,weight_nonbio=feat
                elif cfg.MODEL.LAST_LAYER in ['clipFc','clipMLP']:
                    image_features_bio=score['clip_bio_score']    
                    if 'clip_nonbio_score' in score:
                        image_features_nonbio=score['clip_nonbio_score']
                    else:
                        image_features_nonbio=None
                # if 'cip_bio_reverse_score' in score:
                #     image_feature_reverse=score['clip_bio_reverse_score']
                #     #

                image_features_bio=torch.nn.functional.normalize(image_features_bio,dim=-1)
                if image_features_nonbio is not None:
                    image_features_nonbio=torch.nn.functional.normalize(image_features_nonbio,dim=-1)
                if image_features_bio.shape[1]>200:
                    image_features_bio=image_features_bio.unsqueeze(1)
                # if caption_feature.shape[1]>200:
                #     caption_feature=caption_feature.unsqueeze(1)
                        
                caption_feature=caption_feature.type(image_features_bio.type())
                text_features_bio=caption_feature[:,:image_features_bio.shape[1]]
                text_features_bio=torch.nn.functional.normalize(text_features_bio,dim=-1)
                text_features_nonbio=caption_feature[:,image_features_bio.shape[1]:]
                text_features_nonbio=torch.nn.functional.normalize(text_features_nonbio,dim=-1)
                #text_features_bio.norm(dim=-1, keepdim=True)
            

            
            if 'clipBio' in lossTypes:
                loss_clip_all=0
                for i in range(image_features_bio.shape[1]):
                    logits_per_bio = image_features_bio[:,i]  @ text_features_bio[:,i].T
                    labels = torch.arange(image_features_bio[:,i].shape[0],device=image_features_bio.device)
                    loss_clip = F.cross_entropy(logits_per_bio, labels)
                   
                    i2t_acc = (logits_per_bio.argmax(-1) == labels).sum() / len(logits_per_bio)
                   
                        
                    loss_clip_all+=loss_clip
                    train_writer.add_scalar('acc/clip_bio_'+str(i), i2t_acc.item(), step)
                        
                    #loss_clip = loss_image 
                    train_writer.add_scalar('loss/clip_bio_'+str(i), loss_clip.item(), step)
                    #print(loss_clip)
               
                loss+=loss_clip
                
            if 'clipBioReverse' in lossTypes :
                image_feature_reverse=score['clip_bio_reverse_score']
                loss_clip_reverse=0
              
                for i in range(image_feature_reverse.shape[1]):
                    image_feature_reverse_i=torch.nn.functional.normalize(image_feature_reverse[i])
                    image_feature_reverse_i=image_feature_reverse_i.type(text_features_nonbio.type())
                    logits_per_bio_reverse = image_feature_reverse_i @ text_features_nonbio[:,i].T
                    # symmetric loss function
                    labels = torch.arange(image_feature_reverse[i].shape[0],device=image_feature_reverse[i].device)
                    loss_image = F.cross_entropy(logits_per_bio_reverse, labels)
                    
                    i2t_acc = (logits_per_bio_reverse.argmax(-1) == labels).sum() / len(logits_per_bio_reverse)
                    train_writer.add_scalar('acc/clip_bio_reverse_'+str(i), i2t_acc.item(), step)
                    #print(loss_image)
                    loss_clip_reverse += loss_image 
                    train_writer.add_scalar('loss/clip_bio_reverse_'+str(i), loss_clip_reverse.item(), step)      
                loss+=loss_clip_reverse


            return loss

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
            'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


