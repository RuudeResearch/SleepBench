
import torch
from loguru import logger
import os
import sys
sys.path.append('/oak/stanford/groups/jamesz/magnusrk/pretraining_comparison')
from comparison.utils import *
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
import glob
import h5py
from torch.utils.data import Dataset, DataLoader


class SleepEventClassificationDataset(Dataset):
    def __init__(self, 
                 config,
                 channel_groups=None,
                 hdf5_paths=[],
                 split="train",
                 pretrain_type = "MAE",
                 specific_files = None):

        self.config = config
        #self.max_channels = self.config["max_channels"]
        self.context = int(self.config["context"])
        self.channel_like = self.config["channel_like"]

        #diagnosis, death, and demographics
        self.df_demographics = pd.read_csv(config['demographics_labels_path'])
        self.df_diagnosis_presence = pd.read_csv(os.path.join(config['diagnosis_labels_path'], 'is_event.csv'))
        self.df_diagnosis_time = pd.read_csv(os.path.join(config['diagnosis_labels_path'], 'time_to_event.csv'))
        self.df_death_presence = pd.read_csv(os.path.join(config['death_labels_path'], 'is_event.csv'), usecols=['Study ID','death'])
        self.df_death_time = pd.read_csv(os.path.join(config['death_labels_path'], 'time_to_event.csv'), usecols=['Study ID','death'])

        unique_study_ids_in_demo_diag_death = set(self.df_demographics['Study ID'].values).intersection(set(self.df_diagnosis_presence['Study ID'].values)).intersection(set(self.df_diagnosis_time['Study ID'].values)).intersection(set(self.df_death_presence['Study ID'].values)).intersection(set(self.df_death_time['Study ID'].values))

        labels_path = self.config["labels_path"]
        dataset = self.config["dataset"]
        dataset = dataset.split(",")

        label_files = []

        for dataset_name in dataset:
            label_files += glob.glob(os.path.join(labels_path, dataset_name, "**", "*.csv"), recursive=True)

        # label_files = [label_file for label_file in os.listdir(labels_path) if label_file.endswith(".csv")]

        hdf5_paths = load_data(config["split_path"])[split]
        #print(f'first hdf5_paths: {hdf5_paths[0]}')
        #print(f'len hdf5_paths: {len(hdf5_paths)}')
        #print(f'first label_files: {label_files[0]}')
        #print(f'len label_files: {len(label_files)}')
        study_ids = set([os.path.basename(label_file).split(".")[0] for label_file in label_files])
        #print(f'first study_ids: {list(study_ids)[0]}')
        #print(f'len study_ids: {len(study_ids)}')

        hdf5_paths = [f for f in hdf5_paths if os.path.exists(f)]
        #print(f'len hdf5_paths: {len(hdf5_paths)}')
        hdf5_paths = [f for f in hdf5_paths if f.split("/")[-1].split(".")[0] in study_ids]
        hdf5_paths = [f for f in hdf5_paths if f.split("/")[-1].split(".")[0] in unique_study_ids_in_demo_diag_death]
        #print(f'len hdf5_paths: {len(hdf5_paths)}')

        hdf5_paths_ids = set([os.path.basename(hdf5_path).split(".")[0] for hdf5_path in hdf5_paths])
        #print(f'first hdf5_paths_ids: {list(hdf5_paths_ids)[0]}')
        #print(f'len hdf5_paths_ids: {len(hdf5_paths_ids)}')

        hdf5_paths_new = []
        #print(f'dataset: {dataset}')
        #for dataset_name in dataset:
            #hdf5_paths_new += glob.glob(os.path.join(config["embedding_path"], dataset_name, "**", "*.hdf5"), recursive=True)
        if pretrain_type:
            hdf5_paths_new += glob.glob(os.path.join(config["embedding_path"], pretrain_type, "**", "*.hdf5"), recursive=True)
        else:
            hdf5_paths_new += glob.glob(os.path.join(config["embedding_path"], "**", "*.hdf5"), recursive=True)
        print(f'embs_path: {config["embedding_path"]}')
        print(f'first hdf5_paths_new: {hdf5_paths_new[0]}')
        
        #print(f'len hdf5_paths_new: {len(hdf5_paths_new)}')
        
        hdf5_paths_new = [item for item in hdf5_paths_new if os.path.basename(item).split(".")[0] in hdf5_paths_ids]
        #print(f'len hdf5_paths_new: {len(hdf5_paths_new)}')
        hdf5_paths = hdf5_paths_new
        hdf5_paths = [f for f in hdf5_paths if os.path.exists(f)]
        #print(f'len hdf5_paths: {len(hdf5_paths)}')

        if config["max_files"]:
            hdf5_paths = hdf5_paths[:config["max_files"]]
        else:
            hdf5_paths = hdf5_paths

        labels_dict = {
            os.path.basename(item).split(".")[0]: item for item in label_files
        }
        if specific_files:
            #print(f'hdf5_paths[0] {hdf5_paths[0]}')
            #print(f'specific_files[0] {specific_files[0]}')
            
            # Extract base names from specific_files (without extension) for proper comparison
            specific_files_base = [os.path.splitext(f)[0] for f in specific_files]
            
            # Filter hdf5_paths to only include files whose base names are in specific_files
            hdf5_paths = [f for f in hdf5_paths if os.path.splitext(os.path.basename(f))[0] in specific_files_base]
            
            print(f'number of specific_files: {len(hdf5_paths)}')

            repeats = max(1024 // len(specific_files), 1)

            
            # Repeat the hdf5 files
            hdf5_paths = [f for f in hdf5_paths for _ in range(repeats)]
            print(f'number of training items per epoch: {len(hdf5_paths)}')
        if self.context == -1:
            self.index_map = [(path, labels_dict[path.split("/")[-1].split(".")[0]], -1) for path in hdf5_paths]
        else:
            self.index_map = []
            loop = tqdm(hdf5_paths[:], total=len(hdf5_paths), desc=f"Indexing {split} data")
            for hdf5_file_path in loop:
                file_prefix = os.path.basename(hdf5_file_path).split(".")[0]
                with h5py.File(hdf5_file_path, "r") as file:
                    dataset_names = list(file.keys())[:]
                    dataset_name = dataset_names[0]
                    dataset_length = file[dataset_name].shape[0]
                    for i in range(0, dataset_length, self.context):
                        self.index_map.append((hdf5_file_path, labels_dict[file_prefix], i))           
            
        #logger.info(f"Number of files in {split} set: {len(hdf5_paths)}")
        #logger.info(f"Number of files to be processed in {split} set: {len(self.index_map)}")
        self.total_len = len(self.index_map)
        self.max_seq_len = config["model_params"]["max_seq_length"]

    def __len__(self):
        return self.total_len

    def get_index_map(self):
        return self.index_map

    def __getitem__(self, idx):
        hdf5_path, label_path, start_index = self.index_map[idx]
        labels_df = pd.read_csv(label_path)
        y_data = labels_df["StageNumber"].to_numpy()
        if self.context != -1:
            y_data = y_data[start_index:start_index+self.context]
        x_data = []
        with h5py.File(hdf5_path, 'r') as hf:
            dset_names = list(hf.keys())[:]
            for dataset_name in dset_names:
                x_data.append(hf[dataset_name][:])
        x_data = np.array(x_data)
        # Convert x_data to tensor
        x_data = torch.tensor(x_data, dtype=torch.float32)
        y_data = torch.tensor(y_data, dtype=torch.float32)
        min_length = min(x_data.shape[1], len(y_data))
        x_data = x_data[:, :min_length, :].squeeze()
        y_data = y_data[:min_length]
        
        #diagnosis, death, and demographics
        study_id = os.path.basename(hdf5_path).split(".")[0]
        try:
            diagnosis_presence = torch.tensor(self.df_diagnosis_presence[self.df_diagnosis_presence['Study ID'] == study_id].values[0][1:].astype(np.float32))
            diagnosis_time = torch.tensor(self.df_diagnosis_time[self.df_diagnosis_time['Study ID'] == study_id].values[0][1:].astype(np.float32))
            death_presence = torch.tensor(self.df_death_presence[self.df_death_presence['Study ID'] == study_id].values[0][1:].astype(np.float32))
            death_time = torch.tensor(self.df_death_time[self.df_death_time['Study ID'] == study_id].values[0][1:].astype(np.float32))
            age = torch.tensor(self.df_demographics[self.df_demographics['Study ID'] == study_id]['Age at Study Date'].values) / 100
        except:
            print(f'Study ID {study_id} not found in demographics, diagnosis, or death data')

        
        return x_data, y_data, self.max_seq_len, hdf5_path, diagnosis_presence, diagnosis_time, death_presence, death_time, age

def finetune_collate_fn(batch):

    x_data, y_data, max_seq_len_list, hdf5_path_list, diagnosis_presence, diagnosis_time, death_presence, death_time, age = zip(*batch)

    # padding the temporal as in sleep_event_finetune_full_collate_fn
    max_seq_len_temp = max([item.size(0) for item in x_data])
    # Determine the max sequence length for padding
    if max_seq_len_list[0] is None:
        max_seq_len = max_seq_len_temp
    else:
        max_seq_len = min(max_seq_len_temp, max_seq_len_list[0])
    
    padded_x_data = []
    padded_y_data = []
    padded_mask = []
    diagnosis_presence_list = []
    diagnosis_time_list = []
    death_presence_list = []
    death_time_list = []
    age_list = []

    for x_item, y_item, diagnosis_presence_item, diagnosis_time_item, death_presence_item, death_time_item, age_item  in zip(x_data, y_data, diagnosis_presence, diagnosis_time, death_presence, death_time, age):
        # Get the shape of x_item
        s, e = x_item.size()

        s = min(s, max_seq_len)

        # Create a padded tensor and a mask tensor for x_data
        padded_x_item = torch.zeros((max_seq_len, e))
        mask = torch.ones((max_seq_len))

        # Copy the actual data to the padded tensor and set the mask for real data
        padded_x_item[:s, :e] = x_item[:s, :e]
        mask[:s] = 0  # 0 for real data, 1 for padding

        # Pad y_data with zeros to match max_seq_len
        padded_y_item = torch.zeros(max_seq_len)
        padded_y_item[:s] = y_item[:s]

        # Append padded items to lists
        padded_x_data.append(padded_x_item)
        padded_y_data.append(padded_y_item)
        padded_mask.append(mask)
        diagnosis_presence_list.append(diagnosis_presence_item)
        diagnosis_time_list.append(diagnosis_time_item)
        death_presence_list.append(death_presence_item)
        death_time_list.append(death_time_item)
        age_list.append(age_item)



    # Stack all tensors into a batch
    x_data = torch.stack(padded_x_data)
    y_data = torch.stack(padded_y_data)
    padded_mask = torch.stack(padded_mask)

    diagnosis_presence = torch.stack(diagnosis_presence_list)
    diagnosis_time = torch.stack(diagnosis_time_list)
    death_presence = torch.tensor(death_presence_list).unsqueeze(1)
    death_time = torch.tensor(death_time_list).unsqueeze(1)
    age = torch.tensor(age_list).unsqueeze(1)
    
    return x_data, y_data, padded_mask, hdf5_path_list, diagnosis_presence, diagnosis_time, death_presence, death_time, age

def sleep_event_finetune_full_collate_fn(batch):
    x_data, y_data, max_channels_list, max_seq_len_list, hdf5_path_list = zip(*batch)

    num_channels = max(max_channels_list)

    max_seq_len_temp = max([item.size(1) for item in x_data])
    # Determine the max sequence length for padding
    if max_seq_len_list[0] is None:
        max_seq_len = max_seq_len_temp
    else:
        max_seq_len = min(max_seq_len_temp, max_seq_len_list[0])

    padded_x_data = []
    padded_y_data = []
    padded_mask = []

    for x_item, y_item in zip(x_data, y_data):
        # Get the shape of x_item
        c, s, e = x_item.size()
        c = min(c, num_channels)
        s = min(s, max_seq_len)  # Ensure the sequence length doesn't exceed max_seq_len

        # Create a padded tensor and a mask tensor for x_data
        padded_x_item = torch.zeros((num_channels, max_seq_len, e))
        mask = torch.ones((num_channels, max_seq_len))

        # Copy the actual data to the padded tensor and set the mask for real data
        padded_x_item[:c, :s, :e] = x_item[:c, :s, :e]
        mask[:c, :s] = 0  # 0 for real data, 1 for padding

        # Pad y_data with zeros to match max_seq_len
        padded_y_item = torch.zeros(max_seq_len)
        padded_y_item[:s] = y_item[:s]

        # Append padded items to lists
        padded_x_data.append(padded_x_item)
        padded_y_data.append(padded_y_item)
        padded_mask.append(mask)

    # Stack all tensors into a batch
    x_data = torch.stack(padded_x_data)
    y_data = torch.stack(padded_y_data)
    padded_mask = torch.stack(padded_mask)
    
    return x_data, y_data, padded_mask, hdf5_path_list