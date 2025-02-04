#%%
# This script handles feature extraction, model loading, and prediction for whole slide images (WSIs).
# It leverages multiple deep learning and image processing libraries for efficient data handling and inference.

# **Functions Overview:**
# - fix_state_dict: Adjusts the state dictionary format for model loading.
# - get_embedding: Extracts feature embeddings from image tiles using a model.
# - test: Runs inference on slide-level data and computes probabilities.
# - main: Orchestrates data loading, model preparation, and inference execution.

# **Non-Standard Library Descriptions:**
# - openslide: For reading whole slide image formats (e.g., .svs files).
# - gigapath: Custom module likely for specific model architecture.
# - vision_transformer (vits): Implementation of Vision Transformer models.
# - test_datasets: Custom dataset handling module for loading test data.
# - modules: Custom module, possibly containing model architectures or utilities.
# - extract_tissue_new: Custom module for tissue extraction functions.
# - get_slide_path: Retrieves the file path of a given slide ID allowing for transfer from archive.

# The script is designed for distributed GPU inference and includes memory management for handling large datasets.
#%%


#%%
import argparse
import os
import sys
import time
import shutil
import datetime
import time
import math
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score
import openslide
import gigapath
import vision_transformer as vits
import test_datasets as datasets
import modules
import extract_tissue_new
from get_slide_path import get_slide_path
import gc
#%%
# Set the LD_LIBRARY_PATH environment variable
os.environ['LD_LIBRARY_PATH'] = '/home/chad/anaconda3/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

# Set the device to GPU 2
torch.cuda.set_device(2)
torch.cuda.empty_cache()
#%%
def fix_state_dict(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove 'module.' of DataParallel/DistributedDataParallel
        new_state_dict[name] = v
    return new_state_dict

def get_embedding(loader, model, device, args):
    model.eval()
    tensor = torch.zeros(len(loader.dataset), model.ndim).float().to(device)
    with torch.no_grad():
        for i, img in enumerate(loader):
            print('Features: [{}/{}]'.format(i+1, len(loader)), end='\r')
            h = model(img.to(device))
            tensor[i*args.batch_size:i*args.batch_size+img.size(0),:] = h.detach().clone()
    
    print('')
    return tensor

def test(tloader, sloader, tmodel, smodel, device, args):
    # Set model in test mode
    smodel.eval()
    # Initialize probability vector
    probs = torch.FloatTensor(len(sloader))
    # Loop through batches
    with torch.no_grad():
        for i, filename in enumerate(sloader):
            ## Set slide
            tloader.dataset.set_slide(filename)
            ## Get embedding
            print("Getting embeddings")
            h = get_embedding(tloader, tmodel, device, args)
            ## Forward pass slide model
            _, _, output = smodel(h)
            output = F.softmax(output, dim=1)
            ## Clone output to output vector
            probs[i] = output.detach()[:,1].cpu()
            print(f'[{i+1}/{len(sloader)}]')
    
    print('')
    probs = probs.numpy()
    # auc = roc_auc_score(sloader.dfs.target, probs)
    # print(auc)
    return probs

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch_size', default=30, type=int, help='Per-GPU batch-size: number of distinct images loaded on one GPU.')
    parser.add_argument('--outdir', default="/media/hdd1/chad/MSK_Finetuned_ViT_base/sharing/test_dir", type=str, help='Path to checkpoints and output file.')
    parser.add_argument('--outname', default="predictions_gigapath.csv", type=str, help='Name of output file.')
    parser.add_argument('--workers', default=10, type=int, help='Number of data loading workers.')
    parser.add_argument('--tile_checkpoint', type=str, required=True, help='Path to the tile model checkpoint.')
    parser.add_argument('--slide_checkpoint', type=str, required=True, help='Path to the slide model checkpoint.')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to the CSV file containing test cases.')
    parser.add_argument('--gpu', default=2, type=int, help='GPU device ID to use.')
    parser.add_argument('--tmp_file_holding', type=str, help='where to hold the tmp svs file')
    
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    
    final_out = os.path.join(args.outdir, args.outname)
    
    # Load test data
    test_df = pd.read_csv(args.test_csv)
    # Remove duplicates based on the 'slide_id' column
    test_df = test_df.drop_duplicates(subset='slide_id')
    # Reverse the DataFrame
    reversed_test_df = test_df.iloc[::-1]
    # Reset the index
    reversed_test_df = reversed_test_df.reset_index(drop=True)

    # Iterate over the reversed DataFrame
    for index, row in reversed_test_df.iterrows():
        allocated_memory = torch.cuda.memory_allocated()
        reserved_memory = torch.cuda.memory_reserved()
        # print(f"Memory allocated before cache delete: {allocated_memory / (1024 ** 2)} MB")
        # print(f"Memory reserved: {reserved_memory / (1024 ** 2)} MB")
        # gc.collect()
        # # Clear cache after operation
        # print(torch.cuda.memory_summary())
        for var_name in ['slide', 'val_tile_loader', 'val_slide_loader', 'tile_model', 'slide_model']:
            if var_name in locals():
                var = locals()[var_name]
                if isinstance(var, torch.nn.Module):
                    var.cpu()
                del locals()[var_name]
        
        torch.cuda.empty_cache()
        # Check memory usage
        # allocated_memory = torch.cuda.memory_allocated()
        # reserved_memory = torch.cuda.memory_reserved()
        # print(f"Memory allocated after cache deletion: {allocated_memory / (1024 ** 2)} MB")
        # print(f"Memory reserved: {reserved_memory / (1024 ** 2)} MB")

        try:
            if os.path.exists(final_out):
                df = pd.read_csv(final_out)
            else:
                df = pd.DataFrame(columns=['slide', 'slide_path', 'target', 'slide_file_name', 'sub_specialty', 'case_accessionDate', 'case_id_slide', 'part_id', 'block_name', 'slide_barcode', 'slide_id', 'scanner_id', 'Molecular_Block', 'part_block', 'match_status', 'file_path', 'score'])
                df.to_csv(final_out, index=False)
            print(f"Processing slide ID {row['slide_id']}")
            
            # Check if 'inference_time' column exists, if not, add it
            if 'inference_time' not in df.columns:
                df['inference_time'] = None  # or you can set a default value
            
            if row['slide_id'] in df['slide'].values:
                print(f"Slide {row['slide_id']} already processed. Skipping.")
                continue
            
            row['file_path'] = get_slide_path(row['slide_id']).replace('/Aperio', '')
            print(f"File path: {row['file_path']}")
            source_file = str(row['file_path'])
            destination_dir = args.tmp_file_holding
            os.makedirs(destination_dir, exist_ok=True)
            destination_file = os.path.join(destination_dir, os.path.basename(source_file))
            
            if not os.path.exists(destination_file):
                print(f"Transferring {destination_file}")
                shutil.copy(source_file, destination_file)
            else:
                print(f"File {destination_file} already exists. Skipping copy.")
            print(f"File transferred to {destination_file}")
            slide = openslide.OpenSlide(destination_file)
            print(slide)
            base_mpp = extract_tissue_new.slide_base_mpp(slide)
            level, mult = extract_tissue_new.find_level(slide, 0.5, patchsize=224, base_mpp=base_mpp)
            grid = extract_tissue_new.make_sample_grid(slide, patch_size=224, mpp=0.5, mult=4, base_mpp=base_mpp)
            
            data = [{'x': x, 'y': y, 'slide': row['slide_id'], 'level': level, 'mult': mult} for x, y in grid]
            slid_info = [{'slide': row['slide_id'], 'slide_path': destination_file, 'target': 0}]
            
            dft = pd.DataFrame(data)
            
            dfs = pd.DataFrame(slid_info)
            print(dfs.head())
            print(dfs.head())
            print("Column names dfs:", dfs.columns)
            print("Column names test_df:", test_df.columns)
            dfs = pd.merge(dfs, test_df, left_on='slide', right_on='slide_id', how='left')
            
            val_tile_dset, val_slide_loader = datasets.get_test_dataset(dfs=dfs, dft=dft, tilesize=224)
            val_tile_loader = torch.utils.data.DataLoader(val_tile_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
            
            tile_model = gigapath.get_model()
            tile_model.ndim = 1536
            args.ndim = tile_model.ndim
            slide_model = modules.GMA(ndim=args.ndim, dropout=True)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tile_model = tile_model.to(device)
            slide_model = slide_model.to(device)
            
            tile_ch = torch.load(args.tile_checkpoint, map_location=device)
            slide_ch = torch.load(args.slide_checkpoint, map_location=device)
            tile_ch['tile_model'] = fix_state_dict(tile_ch['tile_model'])
            tile_model.load_state_dict(tile_ch['tile_model'])
            slide_model.load_state_dict(slide_ch['slide_model'])
            print('Loaded checkpoints')
    
            probs = test(val_tile_loader, val_slide_loader, tile_model, slide_model, device, args)
            out = val_slide_loader.dfs.copy()
            out['score'] = probs
            if 'inference_time' not in out.columns:
                out['inference_time'] = None 
            out['inference_time'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            if os.path.exists(final_out):
                existing_df = pd.read_csv(final_out)
            else:
                existing_df = pd.DataFrame()
    
            combined_df = pd.concat([existing_df, out], ignore_index=True)
            combined_df.to_csv(final_out, index=False)
            print("Processing case_id_slide column:")
            print(dfs['case_id_slide'])
            

            
            
            if 'slide_path' in out.columns and os.path.isfile(out['slide_path'].iloc[0]):
                try:
                    os.remove(out['slide_path'].iloc[0])
                    print(f"File {out['slide_path'].iloc[0]} deleted successfully.")
                except Exception as e:
                    print(f"Error deleting file {out['slide_path'].iloc[0]}: {e}")
            else:
                print("Column 'slide_path' does not exist or file does not exist.")
            
    
        except Exception as e:
            
            print(f"Error processing slide ID {row['slide_id']}: {e}")
            continue

if __name__ == '__main__':
    main()
    
#%%  
# import torch

# # Check if GPU is available
# if torch.cuda.is_available():
#     # Set the GPU device
#     torch.cuda.set_device(2)
#     print(f"Using GPU: {torch.cuda.get_device_name(2)}")

#     # Perform a basic tensor operation on the GPU
#     try:
#         # Clear cache before starting
#         torch.cuda.empty_cache()

#         # Create a smaller tensor and move it to the GPU
#         tensor = torch.rand(1000, 1000).cuda()
#         # Perform a basic operation
#         with torch.no_grad():
#             result = tensor * tensor

#         # Check memory usage
#         allocated_memory = torch.cuda.memory_allocated()
#         reserved_memory = torch.cuda.memory_reserved()
#         print(f"Memory allocated: {allocated_memory / (1024 ** 2)} MB")
#         print(f"Memory reserved: {reserved_memory / (1024 ** 2)} MB")

#         # Clear cache after operation
#         del tensor
#         del result
#         torch.cuda.empty_cache()
#         # Check memory usage
#         allocated_memory = torch.cuda.memory_allocated()
#         reserved_memory = torch.cuda.memory_reserved()
#         print(f"Memory allocated: {allocated_memory / (1024 ** 2)} MB")
#         print(f"Memory reserved: {reserved_memory / (1024 ** 2)} MB")

#         print("GPU test passed. Tensor operation successful.")
#     except Exception as e:
#         print(f"An error occurred during the GPU test: {e}")
# else:
#     print("GPU is not available.")
# %%

from datetime import datetime

# Example usage
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(current_time)

# %%
