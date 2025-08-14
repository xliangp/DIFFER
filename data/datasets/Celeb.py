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


class Celeb(object):
    """ Celeb-reID-light

    Reference:
        Huang et al. Celebrities-ReID: A Benchmark for Clothes Variation in Long-Term Person Re-Identification. IJCNN, 2019.

    URL: https://naiq.github.io/LTCC_Perosn_ReID.html#
    """
    dataset_dir = 'Celeb-reID'
    def __init__(self, root='data',caption_dir='../CogVLM/basic_demo/CogVLM_results/Celeb-reID',caption_model='EVA02-CLIP-bigE-14',load_sum_ft=False, **kwargs):
        logger = logging.getLogger('EVA-attribure')
        self.logger=logger
        
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')
        self._check_before_run()
        self.non_bio_index=list(map(int,kwargs['nonbio_index'].strip('*').split('*')))
        

        self.load_sum_ft=load_sum_ft
        self.caption_dir =caption_dir+'_'+caption_model+"_textFeature5"
        if self.load_sum_ft:
            self.caption_sum_dir =caption_dir
            self.ft_name='ft_'+caption_model
            
        train, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes = \
            self._process_dir_train(self.train_dir)
        query, gallery, num_test_pids, num_query_imgs, num_gallery_imgs, num_test_clothes = \
            self._process_dir_test(self.query_dir, self.gallery_dir)
        num_total_pids = num_train_pids + num_test_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs
        num_test_imgs = num_query_imgs + num_gallery_imgs 
        num_total_clothes = num_train_clothes + num_test_clothes

        #logger = logging.getLogger('EVA-attribure')
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
            sum_id_file=os.path.join(self.caption_sum_dir,'train_caption_attribute_id_ft3.json')
            with open(sum_id_file,'r') as f:
                sum_id_info=json.load(f)
                
            sum_cloth_file=os.path.join(self.caption_sum_dir,'train_caption_attribute_clothing_ft3.json')
            with open(sum_cloth_file,'r') as f:
                sum_cloth_info=json.load(f)
            first_key, first_value = next(iter(sum_id_info.items()))
            feat_dim=np.asarray(first_value[self.ft_name]).shape[1]
                
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
           
            if self.non_bio_index==[2] and self.load_sum_ft:
                caption_feature=np.zeros((2,feat_dim))
            else:
                caption_path=os.path.join(self.caption_dir,os.path.splitext(img_path[len(self.dataset_dir)+1:])[0]+'.npy')
                if os.path.isfile(caption_path):
                    caption_feature_load=np.load(caption_path)
                    caption_feature=caption_feature_load[[0]+self.non_bio_index]
                else:
                    self.logger.info(f"fail to find file {caption_path}")
                    caption_feature=caption_feature#np.zeros((2,1024))
                       
            if self.load_sum_ft:
                id_ft=np.asarray(sum_id_info[str(pid)][self.ft_name])
                cloth_ft=np.asarray(sum_cloth_info[clothes][self.ft_name])
                caption_feature[0]=id_ft
                if 2 in self.non_bio_index:
                    cloth_index=self.non_bio_index.index(2)
                    caption_feature[1+cloth_index]=cloth_ft
                    
            pid = pid2label[pid]
            data={
                "image_path": img_path,
                "pid": pid,
                "camid": 0,
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
                "camid": 0,
                "clothes_id": clothes_id,
                }
            query_dataset.append(data)
            
            #query_dataset.append((img_path, pid, camid, clothes_id,''))
            #images_info_query.append({'attributes': imgdir2attribute[img_path]})

        for img_path in gallery_img_paths:
            pid, _, camid = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            # camid -= 1 # index starts from 0
            clothes_id = clothes2label[clothes_id]
            data={
                "image_path": img_path,
                "pid": pid,
                "camid": 0,
                "clothes_id": clothes_id,
                }
            gallery_dataset.append(data)
            #gallery_dataset.append((img_path, pid, camid, clothes_id,''))
            #images_info_gallery.append({'attributes': imgdir2attribute[img_path]})
        
        num_imgs_query = len(query_dataset)
        num_imgs_gallery = len(gallery_dataset)

        return query_dataset, gallery_dataset, num_pids, num_imgs_query, num_imgs_gallery, num_clothes

if __name__ =='__main__':
    dataset=Celeb_light('/data/Data/ReIDData')
    print(dataset.num_train_clothes)
    print(dataset.num_test_clothes)