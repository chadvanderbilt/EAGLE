#%%
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

import math
import pdb
import time
#%%

def get_test_dataset( dfs, dft,tilesize=224):
    '''
    dfs has columns:
    - slide
    - slide_path
    - target
    dft has columns:
    - slide
    - x
    - y
    - level
    - mult
    '''
    # dfs = None
    # dft = None
    # Outputs
    tile_dataset_val = inference_tile_dataset(dft, dfs, tilesize)
    slide_loader_val = inference_slide_loader(dft, dfs)
    return tile_dataset_val, slide_loader_val

class inference_slide_loader(object):
    def __init__(self, dft, dfs):
        self.dfs = dfs
    
    def __getitem__(self, index):
        row = self.dfs.iloc[index]
        return row.slide
    
    def __len__(self):
        return len(self.dfs)

class inference_tile_dataset(data.Dataset):
    def __init__(self, dft, dfs, tilesize):
        self.dft = dft
        self.tilesize = tilesize
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))# Using pre-trained DINO
                                         ])
        self.slides = {}
        a = time.time()
        for i, row in dfs.iterrows():
            print(f'Opening slides: {i+1}/{len(dfs)}', end='\r')
            self.slides[row.slide] = CuImage(row.slide_path)
        
        print('')
        print(f'Opened {len(dfs)} slides in {time.time()-a}s')
    
    def set_slide(self, slide):
        self.curr = self.dft[self.dft.slide==slide].reset_index(drop=True)
    
    def __getitem__(self, index):
        row = self.curr.iloc[index]
        size = int(np.round(self.tilesize * row.mult))
        img = Image.fromarray(np.array(self.slides[row.slide].read_region(location=(int(row.x), int(row.y)), size=(size, size), level=int(row.level))))
        if row.mult != 1:
            img = img.resize((self.tilesize, self.tilesize), Image.LANCZOS)
        img = self.transform(img)
        return img
    
    def __len__(self):
        return len(self.curr)
