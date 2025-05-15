import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import h5py
import sys

sys.path.append('../')
from utils import save_data
from models.model import CL
from models.dataset import SetTransformerDataset, collate_fn_MAE
from utils import load_data

#model_base = '/oak/stanford/groups/mignot/projects/SleepBenchTest/pretrain_comparison/output/final_models'
model_base = os.getcwd().split('/pretrain_comparison/comparison/pipeline')[0] + '/pretrain_comparison/output/final_models'
#model_name = "CL"
models_list = [
    #f"{model_name}/20250112_012421_epoch_3_batch_499_val_loss_3.0042924880981445.pt" # CL LOO
    f"CL_pairwise_epochs_36/20250513_142452_epoch_35.pt" # CL_pairwise
    ]
#target_path_base = '/oak/stanford/groups/mignot/projects/SleepBenchTest/pretrain_comparison/output/final_embeddings'
target_path_base = os.getcwd().split('/pretrain_comparison/comparison/pipeline')[0] + '/pretrain_comparison/output/final_embeddings'
splits = ['pretrain', 'train', 'validation', 'test']
# splits = ["test"]

overwrite = True

config_path = os.getcwd().split('/pretrain_comparison/comparison/pipeline')[0] + '/pretrain_comparison/comparison/config/config_CL_LOO.json'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

config["val_size"] = None
# Load data and configuration
channel_groups = load_data(config['channel_groups_path'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

datasets = {
    split: SetTransformerDataset(config, channel_groups, split=split)
    for split in splits
}

dataloaders = {
    split: DataLoader(
        datasets[split],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=False,
        collate_fn=collate_fn_MAE
    )
    for split in splits
}

# Generate embeddings for each model
for model_item in models_list:
    # Initialize model
    model = CL(
        input_size=config["input_size"],
        output_size=config["output_size"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        max_seq_len=config["max_seq_len"],
        dropout=config["dropout"]
    )
    model_path = os.path.join(model_base, model_item)
    print(f"Loading model from: {model_path}")

    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    except:
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()

    # Prepare target directory
    target_path = os.path.join(target_path_base, model_item.split('.')[0])
    os.makedirs(target_path, exist_ok=True)

    # Process each split
    for split in splits:
        print(f"Processing split: {split}")
        dataloader = dataloaders[split]
        loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)

        for i, batch in loop:
            padded_batch, _, file_paths, dset_names_list, chunk_starts = batch
            padded_batch = padded_batch.to(device)

            with torch.no_grad():
                _, embeddings, _ = model(padded_batch)

                for batch_idx, (channel_emb, file_path, chunk_start) in enumerate(zip(embeddings, file_paths, chunk_starts)):
                    subject_id = os.path.basename(file_path).split('.')[0]
                    output_path = os.path.join(target_path, f"{subject_id}.hdf5")

                    # if not overwrite and os.path.exists(output_path):
                    #     # Skip the file if it exists and overwrite is False
                    #     continue

                    # # Delete the file if it exists and overwrite is True
                    # if overwrite and os.path.exists(output_path):
                    #     os.remove(output_path)

                    with h5py.File(output_path, 'a') as hdf5_file:
                        dset_name = 'embedding'
                        chunk_start_correct = chunk_start // (128 * 5)
                        chunk_end = chunk_start_correct + channel_emb.shape[0]

                        if dset_name in hdf5_file:
                            dset = hdf5_file[dset_name]
                            if dset.shape[0] < chunk_end:
                                dset.resize((chunk_end,) + channel_emb.shape[1:])
                            dset[chunk_start_correct:chunk_end] = channel_emb.cpu().numpy()
                        else:
                            hdf5_file.create_dataset(
                                dset_name,
                                data=channel_emb.cpu().numpy(),
                                chunks=(128,) + channel_emb.shape[1:],
                                maxshape=(None,) + channel_emb.shape[1:]
                            )
