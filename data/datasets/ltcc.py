import os
import re
import glob
#import h5py
import random
import math
import logging
import numpy as np
import os.path as osp
from scipy.io import loadmat
from tools.utils import mkdir_if_missing, write_json, read_json
import json

class LTCC(object):
    """ LTCC

    Reference:
        Qian et al. Long-Term Cloth-Changing Person Re-identification. arXiv:2005.12633, 2020.

    URL: https://naiq.github.io/LTCC_Perosn_ReID.html#
    """
    dataset_dir = 'LTCC/LTCC_ReID'
    def __init__(self, root='data',caption_dir='',caption_model='EVA02-CLIP-bigE-14',load_sum_ft=False, **kwargs):
        #EVA02-CLIP-L-14,EVA02-CLIP-bigE-14
        logger = logging.getLogger('EVA-attribure')
        self.logger=logger
        
        self.dataset_dir = osp.join(root, self.dataset_dir)
        #self.caption_model = caption_model
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')
        self._check_before_run()
        self.load_sum_ft=load_sum_ft
        self.bio_index=list(map(int,kwargs['bio_index'].strip('*').split('*')))
        self.non_bio_index=list(map(int,kwargs['nonbio_index'].strip('*').split('*')))
        
        self.caption_dir =os.path.join(caption_dir,caption_model)
        self.ft_name='ft_'+caption_model
        
        train, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes,num_camera = \
            self._process_dir_train(self.train_dir)
        query, gallery, num_test_pids, num_query_imgs, num_gallery_imgs, num_test_clothes = \
            self._process_dir_test(self.query_dir, self.gallery_dir)
      
        num_total_pids = num_train_pids + num_test_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs
        num_test_imgs = num_query_imgs + num_gallery_imgs 
        num_total_clothes = num_train_clothes + num_test_clothes

        #logger = logging.getLogger('EVA-attribure')
        logger.info("=> LTCC loaded")
        logger.info("Dataset statistics:")
        logger.info("  ----------------------------------------")
        logger.info("  subset   | # ids | # images | # clothes")
        logger.info("  ----------------------------------------")
        logger.info("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_clothes))
        logger.info("  test     | {:5d} | {:8d} | {:9d}".format(num_test_pids, num_test_imgs, num_test_clothes))
        logger.info("  query    | {:5d} | {:8d} |".format(num_test_pids, num_query_imgs))
        logger.info("  gallery  | {:5d} | {:8d} |".format(num_test_pids, num_gallery_imgs))
        logger.info("  ----------------------------------------")
        logger.info("  total    | {:5d} | {:8d} | {:9d}".format(num_total_pids, num_total_imgs, num_total_clothes))
        logger.info("  ----------------------------------------")
        #self.logger=logger

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes
        
        self.num_query_imgs = num_query_imgs
        self.num_gallery_imgs = num_gallery_imgs
        self.pid2clothes = pid2clothes
        self.num_camera=num_camera
        self.num_test_pids = num_test_pids
        
       

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir_train(self, dir_path):
        if self.load_sum_ft:
            self.logger.info("loading summery feature")
            sum_id_file=os.path.join(os.path.dirname(self.caption_dir),'train_caption_summary_biometric.json')
            with open(sum_id_file,'r') as f:
                sum_id_info=json.load(f)
                
            # sum_cloth_file=os.path.join(self.caption_sum_dir,'train_caption_summary_clothes.json')
            # with open(sum_cloth_file,'r') as f:
            #     sum_cloth_info=json.load(f)
            #first_key, first_value = next(iter(sum_id_info.items()))
            #feat_dim=np.asarray(first_value[self.ft_name]).shape[1]
            
        caption_path=dir_path.replace(self.dataset_dir,self.caption_dir)+'.npz'
        all_caption = np.load(caption_path, allow_pickle=True)
        all_caption_features = all_caption['data']
        all_caption_files = all_caption['metadata']
        ftNums=int(all_caption_features.shape[0]/all_caption_files.shape[0])
        all_caption_files=list(all_caption_files)
            
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        img_paths.sort()
        pattern1 = re.compile(r'(\d+)_(\d+)_c(\d+)')
        pattern2 = re.compile(r'(\w+)_c')

        pid_container = set()
        clothes_container = set()
        camera_container = set()
        for img_path in img_paths:
            pid, _, camid = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            pid_container.add(pid)
            clothes_container.add(clothes_id)
            camera_container.add(camid)
        
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        camera_container=sorted(camera_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes_id:label for label, clothes_id in enumerate(clothes_container)}
        #camera2label = {camera_id:label for label, camera_id in enumerate(camera_container)}

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)
        num_camera=len(camera_container)

        dataset = []
        pid2clothes = np.zeros((num_pids, num_clothes))
        for img_path in img_paths:
            pid, _, camid = map(int, pattern1.search(img_path).groups())
            clothes = pattern2.search(img_path).group(1)
            camid -= 1 # index starts from 0

            pid = pid2label[pid]
            clothes_id = clothes2label[clothes]
            
            caption_feature_index=all_caption_files.index(osp.basename(img_path)[:-4])
            caption_feature_load=all_caption_features[caption_feature_index*ftNums:(caption_feature_index+1)*ftNums]
            caption_feature=caption_feature_load[self.bio_index+self.non_bio_index]
            
            if False:
                caption_path=os.path.join(self.caption_dir,os.path.splitext(img_path[len(self.dataset_dir)+1:])[0]+'.npy')
                if os.path.isfile(caption_path):
                    caption_feature_loadOg=np.load(caption_path)
                    caption_feature=caption_feature_load[[0]+self.non_bio_index]
                else:
                    #self.logger.info(f"fail to find file {caption_path}")
                    caption_feature=np.zeros((2,512))
                       
            if self.load_sum_ft and 0 in self.bio_index:
                id_ft=np.asarray(sum_id_info[str(pid)][self.ft_name])
                caption_feature[self.bio_index.index(0)]=id_ft
                
            # if self.load_sum_ft and 2 in (self.bio_index+self.non_bio_index):
            #     cloth_ft=np.asarray(sum_cloth_info[clothes][self.ft_name])
            #     cloth_index=(self.bio_index+self.non_bio_index).index(2)
            #     caption_feature[cloth_index]=cloth_ft
            #caption_feature[1]=cloth_ft
                
           
            data={
                "image_path": img_path,
                "pid": pid,
                "camid": camid,
                "clothes_id": clothes_id,
                "caption_feature":caption_feature,
                }
            dataset.append(data)   
            #dataset.append((img_path, pid, camid, clothes_id,caption_feature))
            pid2clothes[pid, clothes_id] = 1
        
        num_imgs = len(dataset)

        return dataset, num_pids, num_imgs, num_clothes, pid2clothes,num_camera

    def _process_dir_test(self, query_path, gallery_path):
        
        # query_caption_path=query_path.replace(self.dataset_dir,self.caption_dir)+'.npz'
        # query_caption = np.load(query_caption_path, allow_pickle=True)
        # query_caption_features = query_caption['data']
        # query_caption_files = query_caption['metadata']
        # ftNums=int(query_caption_features.shape[0]/query_caption_files.shape[0])
        # query_caption_files=list(query_caption_files)
        
        # gallery_caption_path=gallery_path.replace(self.dataset_dir,self.caption_dir)+'.npz'
        # gallery_caption = np.load(gallery_caption_path, allow_pickle=True)
        # gallery_caption_features = gallery_caption['data']
        # gallery_caption_files = gallery_caption['metadata']
        # gallery_caption_files=list(gallery_caption_files)
        
        query_img_paths = glob.glob(osp.join(query_path, '*.png'))
        gallery_img_paths = glob.glob(osp.join(gallery_path, '*.png'))
        query_img_paths.sort()
        gallery_img_paths.sort()
        pattern1 = re.compile(r'(\d+)_(\d+)_c(\d+)')
        pattern2 = re.compile(r'(\w+)_c')

        pid_container = set()
        clothes_container = set()

        for img_path in query_img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            pid_container.add(pid)
            clothes_container.add(clothes_id)
        for img_path in gallery_img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            pid_container.add(pid)
            clothes_container.add(clothes_id)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)

        clothes2label = {clothes_id:label for label, clothes_id in enumerate(clothes_container)}

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        query_dataset = []
        gallery_dataset = []
       
        for img_path in query_img_paths:
            pid, _, camid = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            camid -= 1 # index starts from 0
            clothes_id = clothes2label[clothes_id]
            
            # caption_path=os.path.join(self.caption_dir,os.path.splitext(img_path[len(self.dataset_dir)+1:])[0]+'.npy')
            # if os.path.isfile(caption_path):
            #     caption_feature=np.load(caption_path)
            # else:
            #     logging.info(f"fail to find file {caption_path}")
            #     caption_feature=np.zeros((2,1024))
            
            data={
                "image_path": img_path,
                "pid": pid,
                "camid": camid,
                "clothes_id": clothes_id,
            }
             
            # caption_feature_index=query_caption_files.index(osp.basename(img_path)[:-4])
            # caption_feature_load=query_caption_features[caption_feature_index*ftNums:(caption_feature_index+1)*ftNums]
            # caption_feature=caption_feature_load[self.bio_index+self.non_bio_index]
            # data['caption_feature']=caption_feature
           
            query_dataset.append(data) 
            #query_dataset.append((img_path, pid, camid, clothes_id,caption_feature[:2,:]))
           

        for img_path in gallery_img_paths:
            pid, _, camid = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            camid -= 1 # index starts from 0
            clothes_id = clothes2label[clothes_id]
            
            # caption_path=os.path.join(self.caption_dir,os.path.splitext(img_path[len(self.dataset_dir)+1:])[0]+'.npy')
            # if os.path.isfile(caption_path):
            #     caption_feature=np.load(caption_path)
            # else:
            #     logging.info(f"fail to find file {caption_path}")
            data={
                "image_path": img_path,
                "pid": pid,
                "camid": camid,
                "clothes_id": clothes_id,
                }
            
            # caption_feature_index=gallery_caption_files.index(osp.basename(img_path)[:-4])
            # caption_feature_load=gallery_caption_features[caption_feature_index*ftNums:(caption_feature_index+1)*ftNums]
            # caption_feature=caption_feature_load[self.bio_index+self.non_bio_index]
            # data['caption_feature']=caption_feature
            gallery_dataset.append(data) 
           
           
           
        num_imgs_query = len(query_dataset)
        num_imgs_gallery = len(gallery_dataset)

        return query_dataset, gallery_dataset, num_pids, num_imgs_query, num_imgs_gallery, num_clothes
