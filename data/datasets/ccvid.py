import os
import re
import glob
import h5py
import random
import math
import logging
import numpy as np
import os.path as osp
import json

class CCVID(object):
    """ CCVID

    Reference:
        Gu et al. Clothes-Changing Person Re-identification with RGB Modality Only. In CVPR, 2022.
    """
    def __init__(self, root='/data/datasets/', caption_dir='../../dataset/CogVLM_results/CCVID',caption_model='EVA02-CLIP-bigE-14',load_sum_ft=True, **kwargs):
        
        logger = logging.getLogger('EVA-attribure')
        self.logger=logger
        
        self.root = osp.join(root, 'CCVID/CCVID')
        self.train_path = osp.join(self.root, 'train.txt')
        self.query_path = osp.join(self.root, 'query.txt')
        self.gallery_path = osp.join(self.root, 'gallery.txt')
        self._check_before_run()
        
        self.load_sum_ft=load_sum_ft
        self.non_bio_index=list(map(int,kwargs['nonbio_index'].strip('*').split('*')))
        
        #self.caption_dir =caption_dir+'_'+caption_model+"_textFeatureIDCloth"
        self.caption_dir =caption_dir+'_'+caption_model+"_textFeature5"
        if self.load_sum_ft:
            self.caption_sum_dir =caption_dir
            self.ft_name='ft_'+caption_model
 
        train,num_train_pids, num_train_clothes, pid2clothes, _,camera_num=self._process_data(self.train_path, relabel=True,skip_sample=10)
        num_train_imgs=len(train)
        
        clothes2label = self._clothes2label_test(self.query_path, self.gallery_path)
        query,num_query_pids, num_query_clothes, _, _,_=self._process_data(self.query_path, skip_sample=0,clothes2label=clothes2label)
        num_query_imgs=len(query)
        
        gallery,num_gallery_pids, num_gallery_clothes, _, _,_=self._process_data(self.gallery_path, skip_sample=0,clothes2label=clothes2label)
        num_gallery_imgs=len(gallery)
        

        num_total_pids = num_train_pids + num_gallery_pids
        num_total_clothes = num_train_clothes + len(clothes2label)
        num_total_images = num_train_imgs + num_query_imgs + num_gallery_imgs 

        #logger = logging.getLogger('reid.dataset')
        logger.info("=> CCVID loaded")
        logger.info("Dataset statistics:")
        logger.info("  ---------------------------------------------")
        logger.info("  subset       | # ids | # images | # clothes")
        logger.info("  ---------------------------------------------")
        logger.info("  train        | {:5d} | {:11d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_clothes))
        logger.info("  query        | {:5d} | {:11d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_clothes))
        logger.info("  gallery      | {:5d} | {:11d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_clothes))
        logger.info("  ---------------------------------------------")
        logger.info("  total        | {:5d} | {:11d} | {:9d}".format(num_total_pids, num_total_images, num_total_clothes))
        logger.info("  ---------------------------------------------")

        self.train = train
       
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes
        self.pid2clothes = pid2clothes
        self.num_camera=camera_num
        self.num_query_imgs=num_query_imgs


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_path):
            raise RuntimeError("'{}' is not available".format(self.train_path))
        if not osp.exists(self.query_path):
            raise RuntimeError("'{}' is not available".format(self.query_path))
        if not osp.exists(self.gallery_path):
            raise RuntimeError("'{}' is not available".format(self.gallery_path))

    def _clothes2label_test(self, query_path, gallery_path):
        pid_container = set()
        clothes_container = set()
        with open(query_path, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                tracklet_path, pid, clothes_label = new_line.split()
                clothes = '{}_{}'.format(pid, clothes_label)
                pid_container.add(pid)
                clothes_container.add(clothes)
        with open(gallery_path, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                tracklet_path, pid, clothes_label = new_line.split()
                clothes = '{}_{}'.format(pid, clothes_label)
                pid_container.add(pid)
                clothes_container.add(clothes)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes:label for label, clothes in enumerate(clothes_container)}

        return clothes2label

    def _process_data(self, data_path, relabel=False, clothes2label=None,skip_sample=0):
        tracklet_path_list = []
        pid_container = set()
        clothes_container = set()
        camera_container = set()
        trainFlag=data_path.find('train')>0
        if trainFlag and self.load_sum_ft:
            self.logger.info("loading summery feature")
            sum_id_file=os.path.join(self.caption_sum_dir,'train_caption_attribute_id_ft3.json')
            print(os.path.abspath(sum_id_file))
            with open(sum_id_file,'r') as f:
                sum_id_info=json.load(f)
                
            sum_cloth_file=os.path.join(self.caption_sum_dir,'train_caption_attribute_clothing_ft3.json')
            with open(sum_cloth_file,'r') as f:
                sum_cloth_info=json.load(f)
            first_key, first_value = next(iter(sum_id_info.items()))
            feat_dim=np.asarray(first_value[self.ft_name]).shape[1]
            
        with open(data_path, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                tracklet_path, pid, clothes_label = new_line.split()
                tracklet_path_list.append((tracklet_path, pid, clothes_label))
                clothes = '{}_{}'.format(pid, clothes_label)
                pid_container.add(pid)
                clothes_container.add(clothes)
                
                session = tracklet_path.split('/')[0]
                cam = tracklet_path.split('_')[1]
                if session == 'session3':
                    camid = int(cam) + 12
                else:
                    camid = int(cam)
                camera_container.add(camid)
                
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        camera_container=sorted(camera_container)
        camera_num=len(camera_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        if clothes2label is None:
            clothes2label = {clothes:label for label, clothes in enumerate(clothes_container)}

        
        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        
        pid2clothes = np.zeros((num_pids, len(clothes2label)))

        dataset = []
        for tracklet_path, pid, clothes_label in tracklet_path_list:
            img_paths = glob.glob(osp.join(self.root, tracklet_path, '*')) 
            img_paths.sort()

            clothes = '{}_{}'.format(pid, clothes_label)
            clothes_id = clothes2label[clothes]
            pid2clothes[pid2label[pid], clothes_id] = 1
            if relabel:
                pid_label = pid2label[pid]
            else:
                pid_label = int(pid)
            session = tracklet_path.split('/')[0]
            cam = tracklet_path.split('_')[1]
            if session == 'session3':
                camid = int(cam) + 12
            else:
                camid = int(cam)
            for file in img_paths:
                if skip_sample>0:
                    fileName=os.path.basename(file)
                    try:
                        file_index=int(fileName[:fileName.rfind('.')])
                    except:
                        file_index=0
                    if file_index%skip_sample!=0:
                        continue
                if trainFlag:
                    if self.non_bio_index==[2] and self.load_sum_ft:
                        caption_feature=np.zeros((2,feat_dim))
                    else:
                        caption_path=os.path.join(self.caption_dir,os.path.splitext(file[len(self.root)+1:])[0]+'.npy')
                        if os.path.isfile(caption_path):
                            caption_feature_load=np.load(caption_path)
                            caption_feature=caption_feature_load[[0]+self.non_bio_index]
                        else:
                            #self.logger.info(f"fail to find file {caption_path}")
                            caption_feature=np.zeros((2,512))
                            
                    if self.load_sum_ft:
                        id_ft=np.asarray(sum_id_info[str(pid)][self.ft_name])
                        cloth_ft=np.asarray(sum_cloth_info[clothes][self.ft_name])
                        caption_feature[0]=id_ft
                        if 2 in self.non_bio_index:
                            cloth_index=self.non_bio_index.index(2)
                            caption_feature[1+cloth_index]=cloth_ft
                data={
                    "image_path":  file,
                    "pid": pid_label,
                    "camid": camid,
                    "clothes_id": clothes_id,
                    #"caption_feature":caption_feature,
                    }
                dataset.append(data)  

            #num_imgs_per_tracklet.append(len(img_paths))
            #tracklets.append((img_paths, pid, camid, clothes_id))

        #num_tracklets = len(tracklets)

        return  dataset,num_pids, num_clothes, pid2clothes, clothes2label,camera_num

    def _densesampling_for_trainingset(self, dataset, sampling_step=64):
        ''' Split all videos in training set into lots of clips for dense sampling.

        Args:
            dataset (list): input dataset, each video is organized as (img_paths, pid, camid, clothes_id)
            sampling_step (int): sampling step for dense sampling

        Returns:
            new_dataset (list): output dataset
        '''
        new_dataset = []
        for (img_paths, pid, camid, clothes_id) in dataset:
            if sampling_step != 0:
                num_sampling = len(img_paths)//sampling_step
                if num_sampling == 0:
                    new_dataset.append((img_paths, pid, camid, clothes_id))
                else:
                    for idx in range(num_sampling):
                        if idx == num_sampling - 1:
                            new_dataset.append((img_paths[idx*sampling_step:], pid, camid, clothes_id))
                        else:
                            new_dataset.append((img_paths[idx*sampling_step : (idx+1)*sampling_step], pid, camid, clothes_id))
            else:
                new_dataset.append((img_paths, pid, camid, clothes_id))

        return new_dataset

    def _recombination_for_testset(self, dataset, seq_len=16, stride=64):
        ''' Split all videos in test set into lots of equilong clips.

        Args:
            dataset (list): input dataset, each video is organized as (img_paths, pid, camid, clothes_id)
            seq_len (int): sequence length of each output clip
            stride (int): temporal sampling stride

        Returns:
            new_dataset (list): output dataset with lots of equilong clips
            vid2clip_index (list): a list contains the start and end clip index of each original video
        '''
        new_dataset = []
        vid2clip_index = np.zeros((len(dataset), 2), dtype=int)
        for idx, (img_paths, pid, camid, clothes_id) in enumerate(dataset):
            # start index
            vid2clip_index[idx, 0] = len(new_dataset)
            # process the sequence that can be divisible by seq_len*stride
            for i in range(len(img_paths)//(seq_len*stride)):
                for j in range(stride):
                    begin_idx = i * (seq_len * stride) + j
                    end_idx = (i + 1) * (seq_len * stride)
                    clip_paths = img_paths[begin_idx : end_idx : stride]
                    assert(len(clip_paths) == seq_len)
                    new_dataset.append((clip_paths, pid, camid, clothes_id))
            # process the remaining sequence that can't be divisible by seq_len*stride        
            if len(img_paths)%(seq_len*stride) != 0:
                # reducing stride
                new_stride = (len(img_paths)%(seq_len*stride)) // seq_len
                for i in range(new_stride):
                    begin_idx = len(img_paths) // (seq_len*stride) * (seq_len*stride) + i
                    end_idx = len(img_paths) // (seq_len*stride) * (seq_len*stride) + seq_len * new_stride
                    clip_paths = img_paths[begin_idx : end_idx : new_stride]
                    assert(len(clip_paths) == seq_len)
                    new_dataset.append((clip_paths, pid, camid, clothes_id))
                # process the remaining sequence that can't be divisible by seq_len
                if len(img_paths) % seq_len != 0:
                    clip_paths = img_paths[len(img_paths)//seq_len*seq_len:]
                    # loop padding
                    while len(clip_paths) < seq_len:
                        for index in clip_paths:
                            if len(clip_paths) >= seq_len:
                                break
                            clip_paths.append(index)
                    assert(len(clip_paths) == seq_len)
                    new_dataset.append((clip_paths, pid, camid, clothes_id))
            # end index
            vid2clip_index[idx, 1] = len(new_dataset)
            assert((vid2clip_index[idx, 1]-vid2clip_index[idx, 0]) == math.ceil(len(img_paths)/seq_len))

        return new_dataset, vid2clip_index.tolist()

