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

class LTCC(object):
    """ LTCC

    Reference:
        Qian et al. Long-Term Cloth-Changing Person Re-identification. arXiv:2005.12633, 2020.

    URL: https://naiq.github.io/LTCC_Perosn_ReID.html#
    """
    dataset_dir = 'LTCC/LTCC_ReID'
    def __init__(self, root='data',caption_dir='../CogVLM/basic_demo/CogVLM_results/LTCC_ReID',caption_model='EVA02-CLIP-bigE-14', **kwargs):
        #EVA02-CLIP-L-14,
        self.dataset_dir = osp.join(root, self.dataset_dir)
        #self.caption_model = caption_model
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')
        self._check_before_run()
        
        self.load_sum_ft=True
        self.caption_dir =caption_dir+'_'+caption_model+"_textFeatureIDCloth"
        if self.load_sum_ft:
            self.caption_sum_dir =caption_dir
            self.ft_name='ft_'+caption_model
            
        train, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes,num_camera = \
            self._process_dir_train(self.train_dir)
        query, gallery, num_test_pids, num_query_imgs, num_gallery_imgs, num_test_clothes = \
            self._process_dir_test(self.query_dir, self.gallery_dir)
        num_total_pids = num_train_pids + num_test_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs
        num_test_imgs = num_query_imgs + num_gallery_imgs 
        num_total_clothes = num_train_clothes + num_test_clothes

        logger = logging.getLogger('EVA-attribure')
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

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes
        self.num_query_imgs = num_query_imgs
        self.pid2clothes = pid2clothes
        self.num_camera=num_camera

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
            if self.load_sum_ft:
                id_ft=np.asarray(sum_id_info[str(pid)][self.ft_name])
                cloth_ft=np.asarray(sum_cloth_info[clothes][self.ft_name])
                caption_feature=np.concatenate((id_ft,cloth_ft))
            else:
                caption_path=os.path.join(self.caption_dir,os.path.splitext(img_path[len(self.dataset_dir)+1:])[0]+'.npy')
                os.path.abspath(caption_path)
                if os.path.isfile(caption_path):
                    caption_feature=np.load(caption_path)
                else:
                    logging.info(f"fail to find file {caption_path}")
                    caption_feature=np.zeros((2,512))
                    
            
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
            gallery_dataset.append(data) 
            #gallery_dataset.append((img_path, pid, camid, clothes_id,caption_feature[:2,:]))
           
           
        num_imgs_query = len(query_dataset)
        num_imgs_gallery = len(gallery_dataset)

        return query_dataset, gallery_dataset, num_pids, num_imgs_query, num_imgs_gallery, num_clothes

