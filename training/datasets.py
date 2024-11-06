import PIL.Image as Image
from cucim import CuImage
import sys
import os
import numpy as np
import pandas as pd
import random
import torch
import torch.utils.data as data
from sklearn.metrics import roc_auc_score
import torchvision.transforms as transforms
import extra_transforms
import math
import pdb
import time

class training_tile_dataset_binary(data.Dataset):
    def __init__(self, k, tilesize=224, drop=0, seed=1634, target='EGFR_KD', rank=0):
        '''
        k is total number of tiles: tiles_per_gpu * ngpu
        '''
        dfs = pd.read_csv('slide_data.csv')# File contains slide level information
        dfs = dfs[dfs.split=='train'].reset_index(drop=True)
        dft = pd.read_csv('tile_data.csv')# File contains coordinate information for each slide
        dft = dft[dft.slide.isin(dfs.slide)].reset_index(drop=True)
        dfs['target'] = dfs[target]
        self.master_dfs = dfs
        self.master_dft = dft
        self.rank = rank
        self.k = k
        self.dft = None
        self.dfs = None
        self.nslides = int(len(dfs) * (1-drop))
        self.tilesize = tilesize
        self.seed = seed
        self.transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                             extra_transforms.RandomRectRotation(),
                                             extra_transforms.GaussianBlur(),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                         ])
        self.slides = {}
        a = time.time()
        for i, row in dfs.iterrows():
            print(f'Rank {self.rank} - Opening slides: {i+1}/{len(dfs)}', end='\r')
            self.slides[row.slide] = CuImage(os.path.join(row.path, f'{row.slide}.svs'))
        
        print('')
        print(f'Rank {self.rank} - Opened {len(dfs)} slides in {time.time()-a}s')
    
    def _sample_slides(self):
        self.dfs = self.master_dfs.sample(n=self.nslides).reset_index(drop=True)
    
    def _sample_tile_data(self):
        a = time.time()
        tmps = []
        for i, row in self.dfs.iterrows():
            print(f'Rank {self.rank} - Sampling tiles: {i+1}/{len(self.dfs)}', end='\r')
            tmp = self.master_dft[self.master_dft.slide==row.slide]
            if len(tmp) >= self.k:
                tmp = tmp.sample(n=self.k)
            else:
                tmp = tmp.sample(n=self.k, replace=True)
            
            tmp['slide'] = row.slide
            tmps.append(tmp)
        
        print('')
        self.dft = pd.concat(tmps).reset_index(drop=True)
        print(f'Rank {self.rank} - Sampled {len(self.dft)} tiles in {time.time()-a}s')
    
    def makeData(self, epoch):
        np.random.seed(self.seed+epoch)
        random.seed(self.seed+epoch)
        self._sample_slides()
        self._sample_tile_data()
    
    def get_target(self, batch_index):
        row = self.dfs.iloc[batch_index]
        return int(row.target)
    
    def __len__(self):
        return self.nslides * self.k
    
    def __getitem__(self, index):
        row = self.dft.iloc[index]
        size = int(np.round(self.tilesize * row.mult))
        img = Image.fromarray(np.array(self.slides[row.slide].read_region(location=(int(row.x), int(row.y)), size=(size, size), level=int(row.level))))
        if row.mult != 1:
            img = img.resize((self.tilesize, self.tilesize), Image.LANCZOS)
        img = self.transform(img)
        return img

class MyDistributedBatchSampler(object):
    '''
    Ensure that rank 1->N receive the right data splits
    '''
    def __init__(self, dataset, num_replicas=None, rank=None):
        '''
        num_replicas: processes involved in DDP, N
        rank 0: master process, does not receive data
        rank 1,N+1: receives data splits
        '''
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
    
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.rank == 0:
            indices = indices[self.rank:self.total_size:self.num_replicas]
        else:
            indices = indices[self.rank-1:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples
        
        return iter(indices)
    
    def __len__(self):
        return self.num_samples
