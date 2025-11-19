import os
import re
import glob
import h5py
import random
import math
import logging
import numpy as np
import os.path as osp
from scipy.io import loadmat
from tools.utils import mkdir_if_missing, write_json, read_json
import json


class Celeb_light(object):
    """ Celeb-reID-light

    Reference:
        Huang et al. Celebrities-ReID: A Benchmark for Clothes Variation in Long-Term Person Re-Identification. IJCNN, 2019.

    URL: https://naiq.github.io/LTCC_Perosn_ReID.html#
    """
    dataset_dir = 'Celeb-reID-light'
    def __init__(self, root='data',caption_dir='',caption_model='EVA02-CLIP-bigE-14',load_sum_ft=False, **kwargs):
        logger = logging.getLogger('DIFFER')
        
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')
        self._check_before_run()
        
        self.bio_index=list(map(int,kwargs['bio_index']))
        self.non_bio_index=list(map(int,kwargs['nonbio_index']))
        

        self.load_sum_ft=load_sum_ft
        self.caption_dir =os.path.join(caption_dir,caption_model)
        self.ft_name='ft_'+caption_model
            
        train, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes = \
            self._process_dir_train(self.train_dir)
        query, gallery, num_test_pids, num_query_imgs, num_gallery_imgs, num_test_clothes = \
            self._process_dir_test(self.query_dir, self.gallery_dir)
        num_total_pids = num_train_pids + num_test_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs
        num_test_imgs = num_query_imgs + num_gallery_imgs 
        num_total_clothes = num_train_clothes + num_test_clothes


        logger.info("=> Celeb-light loaded")
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
        self.num_train_clothes = num_train_clothes  # 9021
        self.num_test_clothes = num_test_clothes  # 1821
        self.num_query_imgs = num_query_imgs
        self.pid2clothes = pid2clothes
        self.num_camera=1
        self.num_test_pids = num_test_pids
        self.num_gallery_imgs = num_gallery_imgs

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
            sum_id_file=os.path.join(self.caption_sum_dir,'train_caption_summary_biometric.json')
            with open(sum_id_file,'r') as f:
                sum_id_info=json.load(f)
                
            # sum_cloth_file=os.path.join(self.caption_sum_dir,'train_caption_summary_clothes.json')
            # with open(sum_cloth_file,'r') as f:
            #     sum_cloth_info=json.load(f)
            
            
        caption_path=os.path.join(self.caption_dir,dir_path.split('/')[-1]+'.npz')
        all_caption = np.load(caption_path, allow_pickle=True)
        all_caption_features = all_caption['data']
        all_caption_files = all_caption['metadata']
        ftNums=int(all_caption_features.shape[0]/all_caption_files.shape[0])
        all_caption_files=list(all_caption_files)
                
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        img_paths.sort()
        pattern1 = re.compile(r'(\d+)_(\d+)_(\d+)')
        pattern2 = re.compile(r'(\w+)_')

        pid_container = set()
        clothes_container = set()
        for img_path in img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            pid_container.add(pid)
            clothes_container.add(clothes_id)
        
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes_id:label for label, clothes_id in enumerate(clothes_container)}

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        dataset = []
       
        pid2clothes = np.zeros((num_pids, num_clothes))
       
        for img_path in img_paths:
            pid, _, camid = map(int, pattern1.search(img_path).groups())
            clothes = pattern2.search(img_path).group(1)
            # camid -= 1 # index starts from 0
            
            clothes_id = clothes2label[clothes]
            
            caption_feature_index=all_caption_files.index(img_path[len(dir_path)+1:][:-4])
            caption_feature_load=all_caption_features[caption_feature_index*ftNums:(caption_feature_index+1)*ftNums]
            caption_feature=caption_feature_load[self.bio_index+self.non_bio_index]
              
                    
            pid = pid2label[pid]
            data={
                "image_path": img_path,
                "pid": pid,
                "camid": camid,
                "clothes_id": clothes_id,
                "caption_feature":caption_feature,
                }
            dataset.append(data)
            #dataset.append((img_path, pid, camid, clothes_id,''))
            
          
            pid2clothes[pid, clothes_id] = 1
        
        num_imgs = len(dataset)

        return dataset, num_pids, num_imgs, num_clothes, pid2clothes

    def _process_dir_test(self, query_path, gallery_path):
        query_img_paths = glob.glob(osp.join(query_path, '*.jpg'))
        gallery_img_paths = glob.glob(osp.join(gallery_path, '*.jpg'))
        query_img_paths.sort()
        gallery_img_paths.sort()
        pattern1 = re.compile(r'(\d+)_(\d+)_(\d+)')
        pattern2 = re.compile(r'(\w+)_')

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
       

        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes_id:label for label, clothes_id in enumerate(clothes_container)}

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        query_dataset = []
        gallery_dataset = []
        # images_info_query = []
        # images_info_gallery = []
        for img_path in query_img_paths:
            pid, _, camid = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            # camid -= 1 # index starts from 0
            clothes_id = clothes2label[clothes_id]
            data={
                "image_path": img_path,
                "pid": pid,
                "camid": camid,
                "clothes_id": clothes_id,
                }
            query_dataset.append(data)
            
          
        for img_path in gallery_img_paths:
            pid, _, camid = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            # camid -= 1 # index starts from 0
            clothes_id = clothes2label[clothes_id]
            data={
                "image_path": img_path,
                "pid": pid,
                "camid": camid,
                "clothes_id": clothes_id,
                }
            gallery_dataset.append(data)
            
        num_imgs_query = len(query_dataset)
        num_imgs_gallery = len(gallery_dataset)

        return query_dataset, gallery_dataset, num_pids, num_imgs_query, num_imgs_gallery, num_clothes
