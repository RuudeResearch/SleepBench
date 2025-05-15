import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import wandb
import datetime
import numpy as np

import sys
sys.path.append('../')
from models.model import MAE
from models.dataset import SetTransformerDataset, collate_fn_MAE
from utils import load_data, get_mask
import pickle
import yaml

import h5py

model_base = os.getcwd().split('/pretrain_comparison/comparison/pipeline')[0] + '/pretrain_comparison/output/final_models'
models_list = [#'MAE/20241219_063019_epoch_25_batch_999_val_loss_0.4778906366099482.pt'
                #, 'FFT_min/20241226_122916_epoch_54_batch_999_val_loss_1.0008.pt'
                #, 'noise/20241227_114349_epoch_38_batch_499_val_loss_0.23804798993197354.pt'
                #, 
                'FFT_min_noise/20241229_164340_epoch_41_batch_1499_val_loss_0.9817.pt'
                # , 'CL'
                # , 'JEPA'
                ]
target_path_base = os.getcwd().split('/pretrain_comparison/comparison/pipeline')[0] + '/pretrain_comparison/output/final_embeddings'
splits = ['pretrain', 'train', 'validation', 'test']


# load the config
config_path = os.getcwd().split('/pretrain_comparison/comparison/pipeline')[0] + '/pretrain_comparison/comparison/config/config_embed.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
channel_groups = load_data(config['channel_groups_path'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = {
        split: SetTransformerDataset(config, channel_groups, split=split)
        for split in splits

    }

dataloader = {
            split: torch.utils.data.DataLoader(dataset[split], batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=False, collate_fn=collate_fn_MAE)
            for split in splits
            }



for model_item in models_list:
    model = MAE(input_size=640, output_size=128, mask_ratio = 0.00)
    model_path = os.path.join(model_base, model_item) 
    print(f'model_path {model_path}')

    target_path = os.path.join(target_path_base, model_item.split('.')[0] + '_' + model_item.split('.')[1])
    print(f'target_path {target_path}')
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except: 
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    for split in splits:
        print(f"Split: {split}")
        loop = tqdm(enumerate(dataloader[split]), total=len(dataloader[split]), leave=False)
        for i, batch in loop:
            padded_batch_list, mask_list, file_paths, dset_names_list, chunk_starts = batch
            padded_batch_list = padded_batch_list.to(device)
            with torch.no_grad():  # Add this line to prevent tracking of gradients
                _, embedding, _ = model(padded_batch_list)
                for batch_idx, (channel_emb, file_path, chunk_start) in enumerate(zip(embedding, file_paths, chunk_starts)):
                    subject_id = os.path.basename(file_path).split('.')[0]
                    output_path = os.path.join(target_path, f"{subject_id}.hdf5")
                    with h5py.File(output_path, 'a') as hdf5_file:
                        dset_name = 'embedding'
                        if dset_name in hdf5_file:
                            dset = hdf5_file[dset_name]
                            chunk_start_correct = chunk_start // (128 * 5)
                            chunk_end = chunk_start_correct + channel_emb.shape[0]
                            if dset.shape[0] < chunk_end:
                                dset.resize((chunk_end,) + channel_emb.shape[1:])
                            dset[chunk_start_correct:chunk_end] = channel_emb.cpu().numpy()
                        else:
                            hdf5_file.create_dataset(dset_name, data=channel_emb.cpu().numpy(), chunks=(128,) + channel_emb.shape[1:], maxshape=(None,) + channel_emb.shape[1:])
