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
import sys
from torch.cuda.amp import autocast, GradScaler
sys.path.append('../')
from utils import save_data
from models.model import CL
from models.dataset import SetTransformerDataset, collate_fn_MAE
from utils import load_data, get_losses, initialize_mae_weights
import pickle
import yaml
import numpy as np
import json


def run_iter(padded_batch, num_modalities, model, mode, device, temperature, batch_size, ij):
    # Forward pass with mixed precision
    with autocast():
        emb, _, _ = model(padded_batch)
        loss = 0.
        if mode == "pairwise":
            pairwise_loss = np.zeros((num_modalities, num_modalities), dtype=float)
            correct = np.zeros((num_modalities, num_modalities), dtype=int)
            pairs = np.zeros((num_modalities, num_modalities), dtype=int)

            for i in range(num_modalities ):
                for j in range(i + 1, num_modalities):

                    logits = torch.matmul(emb[i], emb[j].transpose(0, 1)) * torch.exp(temperature)
                    labels = torch.arange(logits.shape[0], device=device)
        
                    l = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
                    loss += l
                    pairwise_loss[i, j] = l.item()
                    if len(logits) != 0:
                        correct[i, j] = (torch.argmax(logits, axis=0) == labels).sum().item()
                    else:
                        correct[i, j] = 0
                    pairs[i, j] = batch_size
                    
                    l = torch.nn.functional.cross_entropy(logits.transpose(0, 1), labels.to(device), reduction="sum")
                    loss += l
                    pairwise_loss[j, i] = l.item()
                    if len(logits) != 0:
                        correct[j, i] = (torch.argmax(logits, axis=1) == labels).sum().item()
                    else:
                        correct[j, i] = 0
                    pairs[j, i] = batch_size
            loss /= len(ij)

        elif mode == "leave_one_out":

            pairwise_loss = np.zeros((num_modalities, 2), dtype=float)
            correct = np.zeros((num_modalities, 2), dtype=int)
            pairs = np.zeros((num_modalities, 2), dtype=int)

            for i in range(num_modalities):
                other_emb = torch.stack([emb[j] for j in list(range(i)) + list(range(i + 1, num_modalities))]).sum(0) / (num_modalities - 1)
                logits = torch.matmul(emb[i], other_emb.transpose(0, 1)) * torch.exp(temperature)
                labels = torch.arange(logits.shape[0], device=device)
        
                l = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
                loss += l
                pairwise_loss[i, 0] = l.item()
                if len(logits) != 0:
                    correct[i, 0] = (torch.argmax(logits, axis=0) == labels).sum().item()
                else:
                    correct[i, 0] = 0
                pairs[i, 0] = batch_size
                
                l = torch.nn.functional.cross_entropy(logits.transpose(0, 1), labels.to(device), reduction="sum")
                loss += l
                pairwise_loss[i, 1] = l.item()
                if len(logits) != 0:
                    correct[i, 1] = (torch.argmax(logits, axis=1) == labels).sum().item()
                else:
                    correct[i, 1] = 0
                pairs[i, 1] = batch_size
            loss /= num_modalities * 2

    return loss, pairwise_loss, correct, pairs


def main_worker(rank, world_size, config_path):

    current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    min_avg_loss_val = float('inf')
    
    # Initialize the process group
    dist.init_process_group(backend='nccl')

    # Get rank and world_size
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Set device for this process
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Initialize wandb for rank 0 only
    if config['use_wandb'] and rank == 0:
        wandb.init(project='pretraining_comparison', 
                  name=f'train_dev_{current_timestamp}', 
                  config=config)

    patience = config['patience']

    input_size = config["input_size"]
    output_size = config["output_size"]
    num_heads = config["num_heads"]
    num_layers = config["num_layers"]
    max_seq_len = config["max_seq_len"]
    dropout = config["dropout"]
    temperature = config["temperature"]
    temperature = torch.nn.parameter.Parameter(torch.as_tensor(temperature))
    modality_types = config["modality_types"]
    mode = config.get("mode", "leave_one_out")
    num_modalities = len(modality_types)
    ij = sum([((i, j), (j, i)) for i in range(num_modalities) for j in range(i + 1, num_modalities)], ())

    # Initialize model
    model = CL(input_size=input_size, 
                    output_size=output_size, 
                    num_heads=num_heads, 
                    num_layers=num_layers, 
                    max_seq_len=max_seq_len, 
                    dropout=dropout)
    model.to(device)
    initialize_mae_weights(model)

    # Wrap model with DDP
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    model = DDP(model, 
                device_ids=[local_rank], 
                output_device=local_rank, 
                find_unused_parameters=True)

    # Define loss function and optimizer
    optimizer = optim.AdamW(model.parameters(), 
                          lr=config['lr'], 
                          weight_decay=config['weight_decay'])

    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()

    # Load dataset
    channel_groups = load_data(config['channel_groups_path'])
    split = 'pretrain'
    dataset = SetTransformerDataset(config, channel_groups, split=split)
    split_val = 'validation'
    dataset_val = SetTransformerDataset(config, channel_groups, split=split_val)

    # Create DistributedSampler and DataLoader
    sampler = DistributedSampler(dataset)
    sampler_val = DistributedSampler(dataset_val)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        sampler=sampler,
        num_workers=config['num_workers'],
        collate_fn=collate_fn_MAE,
        drop_last=(split == 'pretrain')
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=config['batch_size'],
        sampler=sampler_val,
        num_workers=int(config['num_workers'] // 4),
        collate_fn=collate_fn_MAE
    )

    # breakpoint()

    # What is the structure of the data
    # batch B, C, S (Where S is 5 mins: 60 * 128; channels: 16)

    # Create model save directory
    if rank == 0:
        model_save_dir = os.path.join(config['save_path'], f"{config['model']}_{config['mode']}_epochs_{config['epochs']}")
        os.makedirs(model_save_dir, exist_ok=True)
    
    if config['checkpoint_path'] is not None:
        checkpoint = torch.load(config['checkpoint_path'], map_location=device)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0
        # start_epoch = 18

    print(start_epoch)

    # breakpoint()
    
    if rank == 0:
        print(f'number of iterations per epoch: {len(dataloader)}')

    # Training loop
    num_epochs = config['epochs']
    for epoch in range(start_epoch, num_epochs):
        sampler.set_epoch(epoch)
        model.train()

        epoch_loss = 0.0
        accumulated_loss = 0.0
        if mode == "pairwise":
            total_pairwise_loss = np.zeros((num_modalities, num_modalities), dtype=float)
            total_correct = np.zeros((num_modalities, num_modalities), dtype=int)
            total_n = 0
            total_pairs = np.zeros((num_modalities, num_modalities), dtype=int)
        elif mode == "leave_one_out":
            total_pairwise_loss = np.zeros((num_modalities, 2), dtype=float)
            total_correct = np.zeros((num_modalities, 2), dtype=int)
            total_n = 0
            total_pairs = np.zeros((num_modalities, 2), dtype=int)

        if rank == 0:
            loop = tqdm(enumerate(dataloader), 
                       total=len(dataloader), 
                       desc=f"Epoch [{epoch+1}/{num_epochs}]")
        else:
            loop = enumerate(dataloader)

        optimizer.zero_grad()  # Zero gradients at the start of epoch
        
        for batch_idx, batch in loop:
            # Unpack and move data to device
            padded_batch, _, file_paths, dset_names_list, chunk_starts = batch
            padded_batch = padded_batch.to(device, non_blocking=True)

            batch_size = padded_batch.size(0)

            loss, pairwise_loss, correct, pairs = run_iter(
                padded_batch, num_modalities, model, mode, device, temperature, batch_size, ij
            )

            loss /= batch_size  # TODO I do not exactly know what this is. Figure out. 
            scaled_loss = loss / config['accumulation_interval'] # TODO check if this is correct?

            if temperature < 0:
                with torch.no_grad():
                    temperature.fill_(0)
            
            # Backward pass with gradient scaling
            scaler.scale(scaled_loss).backward()

            accumulated_loss += loss.item()
            total_pairwise_loss += pairwise_loss
            total_correct += correct
            total_n += batch_size
            total_pairs += pairs

            # Calculate accuracy and log that as well along with loss
        
            # Determine if we should step the optimizer
            is_accumulation_step = (batch_idx + 1) % config['accumulation_interval'] == 0
            is_last_batch = (batch_idx + 1) == len(dataloader)
            
            if is_accumulation_step or is_last_batch:
                # Calculate average loss over accumulation interval
                steps_in_interval = batch_idx % config['accumulation_interval'] + 1
                acummulations = min(config['accumulation_interval'], steps_in_interval)
                avg_accumulated_loss = accumulated_loss / acummulations

                if mode == "pairwise":
                    # Calculate and log pairwise accuracies
                    pairwise_accuracies = [
                        100 * (total_correct[i, j] + total_correct[j, i]) / 2 / total_pairs[i, j]
                        if total_pairs[i, j] > 0 else 0.0
                        for i in range(len(modality_types))
                        for j in range(i + 1, len(modality_types))
                    ]
                    
                    if rank == 0:
                        loop.set_postfix_str(
                            f"Loss: {avg_accumulated_loss:.5f}; " +
                            "Acc: {}; ".format(" ".join(map("{:.1f}".format, pairwise_accuracies))) +
                            f"Temperature: {temperature.item():.3f}"
                        )
                        if config['use_wandb']:
                            wandb.log({
                                "Pairwise_train_loss": avg_accumulated_loss,
                                "Pairwise_temperature": temperature.item(),
                                "epoch": epoch,
                                "batch": batch_idx,
                                "learning_rate": optimizer.param_groups[0]['lr'],
                                "scale": scaler.get_scale(),
                                **{f"Pairwise_acc_{i}_{j}": 100 * (total_correct[i, j] + total_correct[j, i]) / 2 / total_pairs[i, j]
                                if total_pairs[i, j] > 0 else 0.0
                                for i in range(len(modality_types))
                                for j in range(i + 1, len(modality_types))}
                            })

                elif mode == "leave_one_out":
                    # Calculate and log leave-one-out accuracies
                    leave_one_out_accuracies = [
                        100 * (total_correct[i, 0] + total_correct[i, 1]) / (total_pairs[i, 0] + total_pairs[i, 1])
                        if (total_pairs[i, 0] + total_pairs[i, 1]) > 0 else 0.0
                        for i in range(num_modalities)
                    ]
                    
                    if rank == 0:
                        loop.set_postfix_str(
                            f"Loss: {avg_accumulated_loss:.5f}; " +
                            "Acc: {}; ".format(" ".join(map("{:.1f}".format, leave_one_out_accuracies))) +
                            f"Temperature: {temperature.item():.3f}"
                        )
                        if config['use_wandb']:
                            wandb.log({
                                "LeaveOneOut_train_loss": avg_accumulated_loss,
                                "LeaveOneOut_temperature": temperature.item(),
                                "epoch": epoch,
                                "batch": batch_idx,
                                "learning_rate": optimizer.param_groups[0]['lr'],
                                "scale": scaler.get_scale(),
                                **{f"LeaveOneOut_acc_{i}_0": 100 * total_correct[i, 0] / total_pairs[i, 0]
                                if total_pairs[i, 0] > 0 else 0.0 for i in range(num_modalities)},
                                **{f"LeaveOneOut_acc_{i}_1": 100 * total_correct[i, 1] / total_pairs[i, 1]
                                if total_pairs[i, 1] > 0 else 0.0 for i in range(num_modalities)}
                            })
                
                # Step optimizer with gradient scaling
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Save model checkpoint if needed
                if rank == 0 and (batch_idx + 1) % config['save_iter'] == 0:
                    print(f"Saving model at epoch {epoch} and batch {batch_idx}")
                    model_save_path = os.path.join(
                        model_save_dir,
                        f'{current_timestamp}_epoch_{epoch}_batch_{batch_idx}.pt'
                    )
                    
                    torch.save({
                        'epoch': epoch,
                        'temperature': temperature.item(),
                        'batch': batch_idx,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),  # Save scaler state
                        'loss': avg_accumulated_loss,
                    }, model_save_path)
                #validation
                if (batch_idx + 1) % config['eval_iter'] == 0:
                    print(f"Evaluating model at epoch {epoch} and batch {batch_idx}") if rank == 0 else None
                    model.eval()
                    sampler_val.set_epoch(epoch)  # Set epoch for validation sampler

                    total_loss_val = 0.

                    if mode == "pairwise":
                        total_pairwise_loss_val = np.zeros((num_modalities, num_modalities), dtype=float)
                        total_correct_val = np.zeros((num_modalities, num_modalities), dtype=int)
                        total_n_val = 0
                        total_pairs_val = np.zeros((num_modalities, num_modalities), dtype=int)
                    elif mode == "leave_one_out":
                        total_loss_val = 0.
                        total_pairwise_loss_val = np.zeros((num_modalities, 2), dtype=float)
                        total_correct_val = np.zeros((num_modalities, 2), dtype=int)
                        total_n_val = 0
                        total_pairs_val = np.zeros((num_modalities, 2), dtype=int)

                    local_loss_accumulators = {"val": 0.0}

                    with torch.no_grad():
                        for val_idx, batch_val in tqdm(enumerate(dataloader_val), 
                                                    total=len(dataloader_val), 
                                                    desc=f"Validation loop",
                                                    disable=rank != 0):
                            padded_batch_val, _, file_paths_val, dset_names_list_val, chunk_starts_val = batch_val
                            padded_batch_val = padded_batch_val.to(device, non_blocking=True)

                            val_loss, pairwise_loss, correct, pairs = run_iter(padded_batch_val, num_modalities, model, mode, device, temperature, batch_size, ij)
                            total_loss_val += val_loss.item()
                            total_pairwise_loss_val += pairwise_loss
                            total_correct_val += correct
                            total_n_val += padded_batch_val.size(0)
                            total_pairs_val += pairs

                        avg_val_loss = total_loss_val / total_n_val

                        if mode == "pairwise":
                            # Calculate pairwise validation accuracies
                            accuracies_val_pairwise = [
                                100 * (total_correct_val[i, j] + total_correct_val[j, i]) / 2 / total_pairs_val[i, j]
                                if total_pairs_val[i, j] > 0 else 0.0
                                for i in range(len(modality_types))
                                for j in range(i + 1, len(modality_types))
                            ]

                            # Log pairwise validation metrics
                            if rank == 0:
                                print(f"Validation Loss: {avg_val_loss:.4f}")
                                print("Validation Pairwise Acc: {}".format(" ".join(map("{:.1f}".format, accuracies_val_pairwise))))

                                if config['use_wandb']:
                                    wandb.log({
                                        'val_loss': avg_val_loss,
                                        'epoch': epoch,
                                        'batch': batch_idx,
                                        **{f"Pairwise_val_acc_{i}_{j}": 100 * (total_correct_val[i, j] + total_correct_val[j, i]) / 2 / total_pairs_val[i, j]
                                        if total_pairs_val[i, j] > 0 else 0.0
                                        for i in range(len(modality_types))
                                        for j in range(i + 1, len(modality_types))}
                                    })

                        elif mode == "leave_one_out":
                            # Calculate leave-one-out validation accuracies
                            accuracies_val_leave_one_out = [
                                100 * (total_correct_val[i, 0] + total_correct_val[i, 1]) / (total_pairs_val[i, 0] + total_pairs_val[i, 1])
                                if (total_pairs_val[i, 0] + total_pairs_val[i, 1]) > 0 else 0.0
                                for i in range(num_modalities)
                            ]

                            # Log leave-one-out validation metrics
                            if rank == 0:
                                print(f"Validation Loss: {avg_val_loss:.4f}")
                                print("Validation Leave-One-Out Acc: {}".format(" ".join(map("{:.1f}".format, accuracies_val_leave_one_out))))

                                if config['use_wandb']:
                                    wandb.log({
                                        'val_loss': avg_val_loss,
                                        'epoch': epoch,
                                        'batch': batch_idx,
                                        **{f"LeaveOneOut_val_acc_{i}_0": 100 * total_correct_val[i, 0] / total_pairs_val[i, 0]
                                        if total_pairs_val[i, 0] > 0 else 0.0 for i in range(num_modalities)},
                                        **{f"LeaveOneOut_val_acc_{i}_1": 100 * total_correct_val[i, 1] / total_pairs_val[i, 1]
                                        if total_pairs_val[i, 1] > 0 else 0.0 for i in range(num_modalities)}
                                    })


                        # Handle early stopping
                        stop_signal = torch.tensor(0, device=device)
                        # if rank == 0:
                        #     if min_avg_loss_val > val_loss:
                        #         min_avg_loss_val = val_loss
                        #         print(f"Saving model with validation loss {val_loss}")
                        #         model_save_path_best_val = os.path.join(
                        #             model_save_dir,
                        #             f'{current_timestamp}_epoch_{epoch}_batch_{batch_idx}_val_loss_{min_avg_loss_val}.pt'
                        #         )
                        #         torch.save({
                        #             'epoch': epoch,
                        #             'temperature': temperature.item(),
                        #             'batch': batch_idx,
                        #             'model_state_dict': model.module.state_dict(),
                        #             'optimizer_state_dict': optimizer.state_dict(),
                        #             'scaler_state_dict': scaler.state_dict(),  # Save scaler state
                        #             'loss': avg_accumulated_loss,
                        #         }, model_save_path_best_val)
                        #         patience = config['patience']
                        #     else:
                        #         patience -= 1
                        #         if patience == 0:
                        #             print(f"Early stopping triggered")
                        #             if config['use_wandb']:
                        #                 wandb.finish()
                        #             stop_signal.fill_(1)

                        # Synchronize before and after broadcast
                        dist.barrier()
                        dist.broadcast(stop_signal, src=0)
                        dist.barrier()

                        if stop_signal.item() == 1:
                            dist.destroy_process_group()
                            return

                        model.train()

                # Reset accumulated loss after stepping
                accumulated_loss = 0.0
            
            epoch_loss += loss.item()

        # Calculate and log epoch statistics
        if rank == 0:
            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_epoch_loss:.4f}")
            
            if config['use_wandb']:
                wandb.log({
                    'epoch_loss': avg_epoch_loss,
                    'epoch': epoch,
                })
            
            # Save epoch checkpoint
            model_save_path = os.path.join(
                model_save_dir,
                f'{current_timestamp}_epoch_{epoch}.pt'
            )
            torch.save({
                'epoch': epoch,
                'temperature': temperature.item(),
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),  # Save scaler state
                'loss': avg_epoch_loss,
            }, model_save_path)

            save_data(config, os.path.join(model_save_dir, "config.json"))
    dist.destroy_process_group()