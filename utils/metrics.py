import logging

import torch
import numpy as np
import os
from tools.eval_metrics import evaluate_with_clothes

from utils.reranking import re_ranking


def compute_ap_cmc(index, good_index, junk_index):
    """ Compute AP and CMC for each sample
    """
    ap = 0
    cmc = np.zeros(len(index))

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1.0
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        ap = ap + d_recall * precision

    return ap, cmc

def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50,ignore_cam=False):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        if ignore_cam:
            orig_cmc = matches[q_idx]
        else:
            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]  # select one row
            
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)

            # compute cmc curve
            # binary vector, positions with value 1 are correct matches
            orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def eval_func_LTCC(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_AP = []
    num_valid_q = 0.0  # number of valid query

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        q_clothid = q_clothes_ids[q_idx]

        order = indices[q_idx]
        # CC
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        remove = remove | ((g_pids[order] == q_pid) & (
                    g_clothes_ids[order] == q_clothid))
        # SC
        # remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        # remove = remove | ((g_pids[order] == q_pid) & ~(g_clothes_ids[order] == q_clothid))

        keep = np.invert(remove)

        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.0

        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


class R1_mAP_eval():
    def __init__(self, num_query, num_gallery=0,max_rank=50, feat_norm=True, reranking=False,logResults=False,ignore_cam=False,gallery_first=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.num_gallery = num_gallery
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.ignore_cam=ignore_cam
        self.gallery_first=gallery_first

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.imgpath=[]

    def update(self, output):  # called once for each batch
        feat, pid, camid, cloth_id,paths=output
        #feat, pid, camid,paths = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.imgpath+=paths

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        if self.gallery_first:
            gf = feats[:self.num_gallery]
            g_pids = np.asarray(self.pids[:self.num_gallery])
            g_camids = np.asarray(self.camids[:self.num_gallery])
            # query
            qf = feats[self.num_gallery:]
            q_pids = np.asarray(self.pids[self.num_gallery:])
            q_camids = np.asarray(self.camids[self.num_gallery:])
        else:
            # query
            qf = feats[:self.num_query]
            q_pids = np.asarray(self.pids[:self.num_query])
            q_camids = np.asarray(self.camids[:self.num_query])
            # gallery
            gf = feats[self.num_query:]
            g_pids = np.asarray(self.pids[self.num_query:])
            g_camids = np.asarray(self.camids[self.num_query:])
            
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance',qf.shape,gf.shape)
            distmat = euclidean_distance(qf, gf)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids,ignore_cam=self.ignore_cam)

        return cmc, mAP, distmat, self.pids, self.camids, qf, gf,self.imgpath

class R1_mAP_eval_LTCC():
    def __init__(self, num_query, num_gallery=0, max_rank=50, feat_norm=True, reranking=False,gallery_first=False):
        super(R1_mAP_eval_LTCC, self).__init__()
        self.num_query = num_query
        self.num_gallery = num_gallery
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.gallery_first=gallery_first

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.cloth_ids = []
        self.imgpath=[]

    def update(self, output):  # called once for each batch
        feat, pid, camid, cloth_id,paths = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.cloth_ids.extend(np.asarray(cloth_id))
        self.imgpath+=paths

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        if self.gallery_first:
            gf = feats[:self.num_gallery]
            g_pids = np.asarray(self.pids[:self.num_gallery])
            g_camids = np.asarray(self.camids[:self.num_gallery])
            g_clothes_ids = np.asarray(self.cloth_ids[:self.num_gallery])
            
            qf = feats[self.num_gallery:]
            q_pids = np.asarray(self.pids[self.num_gallery:])
            q_camids = np.asarray(self.camids[self.num_gallery:])
            q_clothes_ids = np.asarray(self.cloth_ids[self.num_gallery:])   
        else:
            # query
            qf = feats[:self.num_query]
            q_pids = np.asarray(self.pids[:self.num_query])
            q_camids = np.asarray(self.camids[:self.num_query])
            q_clothes_ids = np.asarray(self.cloth_ids[:self.num_query])
            # gallery
            gf = feats[self.num_query:]
            g_pids = np.asarray(self.pids[self.num_query:])
            g_camids = np.asarray(self.camids[self.num_query:])
            g_clothes_ids = np.asarray(self.cloth_ids[self.num_query:])
            
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance',qf.shape,gf.shape)
            distmat = euclidean_distance(qf, gf)

        cmc, mAP = eval_func_LTCC(distmat, q_pids, g_pids, q_camids, g_camids,q_clothes_ids,g_clothes_ids)

        return cmc, mAP, distmat, self.pids, self.camids, qf, gf,self.imgpath

class R1_mAP_eval_CCVID_VIDEO(R1_mAP_eval_LTCC):
    def __init__(self,num_query, max_rank=50, feat_norm=True, reranking=False,query_vid2clip_index=None,gallery_vid2clip_index=None,num_gallery=0):
        # Pass all arguments to the parent class
        super().__init__(num_query, max_rank, feat_norm, reranking)
        self.query_vid2clip_index=query_vid2clip_index
        self.gallery_vid2clip_index=gallery_vid2clip_index
        self.num_gallery=num_gallery
        
    def average_query_features(self,clip_features,clip_pids,clip_camids,clip_clothes_ids,vid2clip_index):
        features=torch.zeros(len(vid2clip_index),clip_features.size(1))
        pids=torch.zeros(len(vid2clip_index))
        camids=torch.zeros(len(vid2clip_index))
        clothes_ids=torch.zeros(len(vid2clip_index))
        imgseqs=[[] for i in range(len(vid2clip_index))]
        for i, idx in enumerate(vid2clip_index):
            #print(cc)
            features[i]=clip_features[idx[0]:idx[1],:].mean(0)
            #features[i]=F.normalize(features[i],p=2,dim=0)
            pids[i]=clip_pids[idx[0]]
            camids[i]=clip_camids[idx[0]]
            clothes_ids[i]=clip_clothes_ids[idx[0]]
            
            imgseqs[i]=os.path.dirname(self.imgpath[idx[0]])
        return features,pids,camids,clothes_ids,imgseqs
    
    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        
        print('=> feats shape:',feats.shape)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        if feats.shape[0]==self.num_query+self.num_gallery:
            # query
            qf = feats[:self.num_query]
            q_pids = np.asarray(self.pids[:self.num_query])
            q_camids = np.asarray(self.camids[:self.num_query])
            q_clothes_ids = np.asarray(self.cloth_ids[:self.num_query])
            # gallery
            gf = feats[self.num_query:]
            g_pids = np.asarray(self.pids[self.num_query:])

            g_camids = np.asarray(self.camids[self.num_query:])
            g_clothes_ids = np.asarray(self.cloth_ids[self.num_query:])
        else:
            print('=> Use the averaged feature of all clips ')
            query_clip_num=self.query_vid2clip_index[-1][-1]
            qf,q_pids,q_camids,q_clothes_ids,q_seq=self.average_query_features(feats[:query_clip_num],
                                                                         self.pids[:query_clip_num],
                                                                         self.camids[:query_clip_num],
                                                                         self.cloth_ids[:query_clip_num],
                                                                         self.query_vid2clip_index)
            gallery_clip_num=self.gallery_vid2clip_index[-1][-1]
            gf,g_pids,g_camids,g_clothes_ids,g_seq=self.average_query_features(feats[query_clip_num:],
                                                                         self.pids[query_clip_num:],
                                                                         self.camids[query_clip_num:],
                                                                         self.cloth_ids[query_clip_num:],
                                                                         self.gallery_vid2clip_index)
            self.imgpath=q_seq+g_seq
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)
        print('=> DistMat shape:',distmat.shape)
        #cmc, mAP = eval_func_LTCC(distmat, q_pids, g_pids, q_camids, g_camids,q_clothes_ids,g_clothes_ids)
        
        cmc, mAP = evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='SC')
        
        
        cmc_cc, mAP_cc=evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='CC')
        

        return cmc_cc, mAP_cc,cmc, mAP, distmat, qf, gf,self.imgpath

