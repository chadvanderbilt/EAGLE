import datasets
import modules
import utils
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import gigapath
import numpy as np
import pandas as pd
import time
from sklearn.metrics import roc_auc_score
import pdb
import random

def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier()

parser = argparse.ArgumentParser()

#I/O PARAMS
parser.add_argument('--outdir', type=str, default='.', help='name of output directory')
parser.add_argument('--outname', type=str, default='convergence.csv', help='name of output convergence file')

#MODEL PARAMS
parser.add_argument('--tilesize', default=224, type=int, help='tile size (default: 224)')
parser.add_argument('--k_per_gpu', default=600, type=int, help='k tiles sampled at training time  for each gpu (default: 600)')
parser.add_argument('--drop', default=0., type=float, help='fraction of slides to drop per epoch (default: 0.)')
parser.add_argument('--target', default='EGFR_KD', choices=['EGFR_KD'], type=str, help='which target to select (default: EGFR_KD)')
parser.add_argument('--pos_weight', default=0.5, type=float, help='unbalanced positive class weight (default: 0.5, balanced classes)')

#OPTIMIZATION PARAMS
parser.add_argument('--optimizer', default='sgd', type=str, help='The optimizer to use (default: sgd)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of linear warmup (highest LR used during training).""")
parser.add_argument('--lr_end', type=float, default=1e-8, help="""Target LR at the end of optimization. We use a cosine LR schedule with linear warmup.""")
parser.add_argument("--warmup_epochs", default=10, type=int, help="Number of epochs for the linear learning-rate warm up.")
parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the weight decay. With ViT, a smaller value at the beginning of training works well.""")
parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the weight decay. We use a cosine schedule for WD and using a larger decay by the end of training improves performance for ViTs.""")
parser.add_argument('--use_amp', type=int, default=0, choices=[0,1], help='to use AMP (default: 0=False).')
#TRAINING PARAMS
parser.add_argument('--nepochs', type=int, default=40, help='number of epochs (default: 40)')
parser.add_argument('--workers', default=10, type=int, help='number of data loading workers (default: 10)')
parser.add_argument('--save_freq', default=2, type=int, help='checkpoint freq (default: 2)')

#DDP
parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html""")

def pseudo_loss(f, fgrad):
    return (f * fgrad).sum()

def encoder_main(args):
    
    # DDP group all ranks except for rank 0
    ddp_group = dist.new_group(ranks=list(range(1,args.world_size)))
    
    # Set data
    dset = datasets.training_tile_dataset_binary(args.k_per_gpu*(args.world_size-1), tilesize=args.tilesize, drop=args.drop, target=args.target, rank=args.rank)
    sampler = datasets.MyDistributedBatchSampler(dset, rank=args.rank, num_replicas=args.world_size-1)
    loader = torch.utils.data.DataLoader(dset, sampler=sampler, batch_size=args.k_per_gpu, shuffle=False, num_workers=args.workers)
    
    # Set models and optimizers
    tile_model = gigapath.get_model()
    tile_model.ndim = 1536
    
    args.ndim = tile_model.ndim
    tile_model.to(args.gpu)
    tile_model = nn.parallel.DistributedDataParallel(tile_model, device_ids=[args.gpu], process_group=ddp_group)
    params_groups = utils.get_params_groups(tile_model)
    if args.optimizer == 'sgd':
        tile_optimizer = optim.SGD(params_groups, lr=0., momentum=args.momentum, dampening=0, nesterov=True)
    elif args.optimizer == 'adam':
        tile_optimizer = optim.Adam(params_groups)
    elif args.optimizer == 'adamw':
        tile_optimizer = optim.AdamW(params_groups)
    
    # Set scaler
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    
    # Resume from checkpoint if available
    start_epoch = utils.restart_from_checkpoint(
        os.path.join(args.outdir, 'checkpoint_tile.pth'),
        tile_model=tile_model,
        tile_optimizer=tile_optimizer,
        scaler=scaler
    )
    
    # Set schedulers
    lr_schedule = utils.cosine_scheduler(
        args.lr,
        args.lr_end,
        args.nepochs,
        len(loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.nepochs,
        len(loader),
    )
    
    # Main training loop
    for epoch in range(start_epoch, args.nepochs+1):
        
        ### Sample training data
        loader.dataset.makeData(epoch)
        
        for i, inputs in enumerate(loader):
            # rank 0 receives no data
            # ranks 1,N+1 receive split data
            
            # Global lr/wd scheduler
            # Update weight decay and learning rate according to their schedule
            it = len(loader) * (epoch-1) + i # global training iteration
            
            for j, param_group in enumerate(tile_optimizer.param_groups):
                param_group["lr"] = lr_schedule[it]
                if j == 0:  # only the first group is regularized
                    param_group["weight_decay"] = wd_schedule[it]
            
            # Forward pass on encoder with autocast and wait
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                inputs = inputs.to(args.gpu)
                features = tile_model.forward(inputs)
                # Make contiguous. Tensors must be contiguous for GPU-GPU communication
                features = features.contiguous()
            
            dist.barrier(group=None)
            features = features.float()
            
            # Send features to aggregator rank
            with torch.no_grad():
                dist.gather(features, None, dst=0, group=None)
            
            # Wait for forward/backward pass on aggregator
            grads = None
            dist.barrier(group=None)
            
            # Receive features from scatter
            grads_recv = torch.zeros((args.k_per_gpu, args.ndim)).to(args.gpu)
            with torch.no_grad():
                dist.scatter(grads_recv, grads, src=0, group=None)
            
            # Generate loss on ddp group
            # Loss has to be scaled by number of DDP processes
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                tile_loss = pseudo_loss(features, grads_recv) * (args.world_size-1)
            
            scaler.scale(tile_loss).backward()
            scaler.step(tile_optimizer)
            scaler.update()
            tile_optimizer.zero_grad()
        
        # End of epoch
        if args.rank == 1:
            obj = {
                'epoch': epoch,
                'tile_model': tile_model.state_dict(),
                'tile_optimizer': tile_optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epoch + 1
            }
            torch.save(obj, os.path.join(args.outdir, 'checkpoint_tile.pth'))
            if epoch % args.save_freq == 0:
                torch.save(obj, os.path.join(args.outdir, f'checkpoint_tile_{epoch:03}.pth'))

def aggregator_main(args):
    
    # DDP group all ranks except for rank 0
    ddp_group = dist.new_group(ranks=list(range(1,args.world_size)))
    
    # Set data
    dset = datasets.training_tile_dataset_binary(args.k_per_gpu*(args.world_size-1), tilesize=args.tilesize, drop=args.drop, target=args.target, rank=args.rank)
    sampler = datasets.MyDistributedBatchSampler(dset, rank=args.rank, num_replicas=args.world_size-1)
    loader = torch.utils.data.DataLoader(dset, sampler=sampler, batch_size=args.k_per_gpu, shuffle=False, num_workers=args.workers)
    
    # Set models and optimizers
    tile_model = gigapath.get_model()
    tile_model.ndim = 1536
    
    args.ndim = tile_model.ndim
    slide_model = modules.GMA(ndim=args.ndim, dropout=True, n_classes=2)
    slide_model.to(args.gpu)
    if args.pos_weight == 0.5:
        criterion = nn.CrossEntropyLoss().to(args.gpu)
    else:
        w1 = torch.Tensor([1-args.pos_weight, args.pos_weight])
        criterion = nn.CrossEntropyLoss(w1).to(args.gpu)
    
    params_groups = utils.get_params_groups(slide_model)
    if args.optimizer == 'sgd':
        slide_optimizer = optim.SGD(params_groups, lr=0., momentum=args.momentum, dampening=0, nesterov=True)
    elif args.optimizer == 'adam':
        slide_optimizer = optim.Adam(params_groups)
    elif args.optimizer == 'adamw':
        slide_optimizer = optim.AdamW(params_groups)
    
    # Resume from checkpoint if available
    start_epoch = utils.restart_from_checkpoint(
        os.path.join(args.outdir, 'checkpoint_slide.pth'),
        slide_model=slide_model,
        slide_optimizer=slide_optimizer
    )
    
    # Set schedulers
    lr_schedule = utils.cosine_scheduler(
        args.lr,
        args.lr_end,
        args.nepochs,
        len(loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.nepochs,
        len(loader),
    )
    
    # Set output files
    outname = os.path.join(args.outdir, args.outname)
    if not os.path.exists(outname) and start_epoch == 1:
        with open(outname, 'w') as fconv:
            fconv.write('epoch,loss\n')
    elif os.path.exists(outname) and start_epoch > 1:
        dummy = pd.read_csv(outname)
        dummy = dummy.tail(1).epoch.item()
        if start_epoch != dummy + 1:
            raise Exception('Wrong convergence and start epoch')
    else:
        raise Exception('Something wrong with convergence file')
    
    # Main training loop
    for epoch in range(start_epoch, args.nepochs+1):
        
        ### Sample training data
        loader.dataset.makeData(epoch)
        
        running_loss = 0.0
        for i, inputs in enumerate(loader):
            # rank 0 receives no data
            # ranks 1,N+1 receive split data
            time0 = time.time()
            
            # Global lr/wd scheduler
            # Update weight decay and learning rate according to their schedule
            it = len(loader) * (epoch-1) + i # global training iteration
            for j, param_group in enumerate(slide_optimizer.param_groups):
                param_group["lr"] = lr_schedule[it]
                if j == 0:  # only the first group is regularized
                    param_group["weight_decay"] = wd_schedule[it]
            
            # Wait for forward pass on encoder
            slide_optimizer.zero_grad()
            features = torch.empty((args.k_per_gpu, args.ndim)).to(args.gpu)
            dist.barrier(group=None)
            
            # Gather features
            with torch.no_grad():
                allfeatures = [torch.empty((args.k_per_gpu, args.ndim)).to(args.gpu) for _ in range(args.world_size)]
                dist.gather(features, allfeatures, dst=0, group=None)
                allfeatures = torch.cat(allfeatures[1:])
            
            # Forward pass on aggregator
            # Record gradients on features
            allfeatures.requires_grad_()
            # Forward / backward for net2
            _, _, output = slide_model(allfeatures)
            label = torch.LongTensor([dset.get_target(i)])
            label = label.to(args.gpu)
            slide_loss = criterion(output, label)
            slide_loss.backward()
            # Get grads of features
            grads = allfeatures.grad.detach()
            # Split grads
            grads = torch.split(grads, args.k_per_gpu)
            grads = [torch.empty((args.k_per_gpu, args.ndim)).to(args.gpu)] + list(grads)
            # Synchronize
            dist.barrier(group=None)
            
            # Define output tensor
            grads_recv = torch.zeros((args.k_per_gpu, args.ndim)).to(args.gpu)
            # Scatter features to other processes
            with torch.no_grad():
                dist.scatter(grads_recv, grads, src=0, group=None)
            
            # Optimizer step
            slide_optimizer.step()
            
            # print statistics
            running_loss += slide_loss.item()
            print(f'{epoch}/{args.nepochs} - [{i+1}/{len(loader)}] - time: {time.time()-time0} - loss: {slide_loss.item():.3f}')
        
        # End of epoch
        running_loss = running_loss / dset.nslides
        with open(outname, 'a') as fconv:
            fconv.write(f'{epoch},{running_loss}\n')
        
        # Model saving logic
        if args.rank == 0:
            obj = {
                'epoch': epoch,
                'slide_model': slide_model.state_dict(),
                'slide_optimizer': slide_optimizer.state_dict(),
                'epoch': epoch + 1
            }
            torch.save(obj, os.path.join(args.outdir, 'checkpoint_slide.pth'))
            if epoch % args.save_freq == 0:
                torch.save(obj, os.path.join(args.outdir, f'checkpoint_slide_{epoch:03}.pth'))

def main():
    # Get user input
    global args
    args = parser.parse_args()
    args.use_amp = bool(args.use_amp)
    init_distributed_mode(args)
    if args.rank == 0:
        aggregator_main(args)
    else:
        encoder_main(args)

if __name__ == '__main__':
    main()
