"""PSG dataset: 5-minute chunks of the 16 harmonized channels from the SSC HDF5 files."""
from torch.utils.data import Dataset
import numpy as np
import h5py
import random
import torch
from tqdm import tqdm
import multiprocessing

from .utils import load_data

def index_file_helper_new(args):
    file_path, channel_like, chunk_size, channel_groups, modality_types = args
    file_index_map = []
    modality_to_channels = {modality_type: [] for modality_type in modality_types}
    try:
        with h5py.File(file_path, 'r', rdcc_nbytes = 300 * 512 * 8 * 2) as hf:
            dset_names = []
            for dset_name in hf.keys():
                if not channel_like or dset_name in channel_like:
                    if isinstance(hf[dset_name], h5py.Dataset):
                        dset_names.append(dset_name)
                        if dset_name in channel_groups["BAS"]:
                            modality_to_channels["BAS"].append(dset_name)
                        if dset_name in channel_groups["RESP"]:
                            modality_to_channels["RESP"].append(dset_name)
                        if dset_name in channel_groups["EKG"]:
                            modality_to_channels["EKG"].append(dset_name)
                        if dset_name in channel_groups["EMG"]:
                            modality_to_channels["EMG"].append(dset_name)
            flag = True
            for modality, channels in modality_to_channels.items():
                if len(channels) == 0:
                    flag = False
                    break
            if flag:
                num_samples = hf[dset_name].shape[0]
                num_chunks = num_samples // chunk_size
                for chunk_start in range(0, num_chunks * chunk_size, chunk_size):
                    file_index_map.append((file_path, dset_names, chunk_start))
    except (OSError, AttributeError) as e:
        with open("problem_hdf5.txt", "a") as f:
            f.write(f"Error processing file {file_path}: {str(e)}\n")
    return file_index_map

def index_files_parallel_new(hdf5_paths, channel_like, samples_per_chunk, num_workers, channel_groups=None, modality_types=None):
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(index_file_helper_new, [(path, channel_like, samples_per_chunk, channel_groups, modality_types) for path in hdf5_paths]), total=len(hdf5_paths), desc="Indexing files", position=0, leave=True))
    return [item for sublist in results for item in sublist]


class SetTransformerDataset(Dataset):
    def __init__(self, 
                 config,
                 channel_groups,
                 hdf5_paths=[],
                 split="pretrain"):

        self.config = config
        self.channel_groups = channel_groups
        channel_like = []
        for modality_type in config["modality_types"]:
            channel_like += channel_groups[modality_type]
        channel_like = set(channel_like)

        if len(hdf5_paths) == 0:
            hdf5_paths = load_data(config["split_path"])[split]

        random.shuffle(hdf5_paths)
        if config["max_files"]:
            self.hdf5_paths = hdf5_paths[:config["max_files"]]
        else:
            self.hdf5_paths = hdf5_paths

        if split == "validation":
            self.hdf5_paths = self.hdf5_paths[:config["val_size"]]
        
        self.samples_per_chunk = config["sampling_duration"] * 60 * config["sampling_freq"]  
        
        # Use multiprocessing to index files in parallel
        self.index_map = index_files_parallel_new(self.hdf5_paths, channel_like, self.samples_per_chunk, config["num_workers"], channel_groups=self.channel_groups, modality_types=config["modality_types"])

        # random.shuffle(self.hdf5_paths)
        # self.index_map = sorted(self.index_map, key=lambda x: (x[1], x[2]))
        self.total_len = len(self.index_map)
        self.modalities_length = []
        for modality_type in self.config["modality_types"]:
            self.modalities_length.append(self.config[f'{modality_type}_CHANNELS'])

    def __len__(self):
        return self.total_len


    def __getitem__(self, idx):
        file_path, original_dset_names, chunk_start = self.index_map[idx]
        
        modality_to_channels = {modality_type: [] for modality_type in self.config["modality_types"]}
        for dset_name in original_dset_names:
            if dset_name in self.channel_groups["BAS"]:
                modality_to_channels["BAS"].append(dset_name)
            if dset_name in self.channel_groups["RESP"]:
                modality_to_channels["RESP"].append(dset_name)
            if dset_name in self.channel_groups["EKG"]:
                modality_to_channels["EKG"].append(dset_name)
            if dset_name in self.channel_groups["EMG"]:
                modality_to_channels["EMG"].append(dset_name)

        target = []
        processed_dset_names = []  # Keep track of channels in processing order
        
        with h5py.File(file_path, 'r', rdcc_nbytes=300 * 512 * 8 * 2) as hf:
            for modality_type in self.config["modality_types"]:
                ds_names = modality_to_channels[modality_type]
                processed_dset_names.extend(ds_names)  # Add channels in processing order
                
                data = np.zeros((len(ds_names), self.samples_per_chunk))
                for idx, ds_name in enumerate(ds_names):
                    signal = hf[ds_name][chunk_start:chunk_start+self.samples_per_chunk]
                    data[idx] = signal
                target.append(torch.from_numpy(data).float())

        return target, file_path, processed_dset_names, chunk_start, self.modalities_length

def collate_fn_MAE(batch):
   file_paths = [item[1] for item in batch]
   dset_names_list = [item[2] for item in batch] 
   chunk_starts = [item[3] for item in batch]
   
   # Extract batch data and combine modalities for each item
   batch_data = [torch.cat(item[0], dim=0) for item in batch]
   # Stack all items in batch
   batched_tensor = torch.stack(batch_data, dim=0)
   
   mask = torch.ones_like(batched_tensor, dtype=torch.bool)
   
   return batched_tensor, mask, file_paths, dset_names_list, chunk_starts


    # def __getitem__(self, idx):
    #     hdf5_path, y_data = self.index_map[idx]

    #     x_data = []
    #     with h5py.File(hdf5_path, 'r') as hf:
    #         dset_names = []
    #         for dset_name in hf.keys():
    #             if isinstance(hf[dset_name], h5py.Dataset):
    #                 dset_names.append(dset_name)

    #         dset_names_new = []
    #         for modality_type in self.config["modality_types"]:
    #             max_modality_channels = self.config[f"{modality_type}_CHANNELS"]
    #             modality_channels = list(set(self.channel_groups[modality_type]) & set(dset_names))
    #             if len(modality_channels) > max_modality_channels:
    #                 modality_channels = random.sample(modality_channels, max_modality_channels)
                
    #             dset_names_new.extend(modality_channels)
            
    #         for dataset_name in dset_names_new:
    #             x_data.append(hf[dataset_name][:])

    #     if not x_data:
    #         # Skip this data point if x_data is empty
    #         return self.__getitem__((idx + 1) % self.total_len)

    #     # Convert x_data list to a single numpy array
    #     x_data = np.array(x_data)

    #     # Convert x_data to tensor
    #     x_data = torch.tensor(x_data, dtype=torch.float32)

    #     # Convert y_data to tensor
    #     y_data = torch.tensor(y_data, dtype=torch.float32)

    #     return x_data, y_data, self.max_channels, self.max_seq_len, hdf5_path


