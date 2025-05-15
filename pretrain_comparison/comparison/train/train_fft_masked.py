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
from models.model import MAE
from models.dataset import SetTransformerDataset, collate_fn_MAE
from utils import load_data, get_losses_fft, initialize_mae_weights
import pickle
import yaml

def main_worker(rank, world_size, config_path):
    # Initialize timestamp and tracking variables
    current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    min_avg_loss_val = float('inf')
    
    # Initialize the distributed process group
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Set up the device
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Initialize wandb for rank 0 only
    if config['use_wandb'] and rank == 0:
        wandb.init(project='pretraining_comparison', 
                  name=f'train_fft_{current_timestamp}', 
                  config=config)

    patience = config['patience']

    # Initialize and set up the model
    model = MAE(mask_ratio=config['mask_ratio'], input_size=640, output_size=128)
    model.to(device)
    initialize_mae_weights(model)

    # Set up distributed training
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    model = DDP(model, 
                device_ids=[local_rank], 
                output_device=local_rank, 
                find_unused_parameters=True)

    # Initialize loss function, optimizer and gradient scaler
    criterion = nn.MSELoss(reduction='none').to(device)
    optimizer = optim.AdamW(model.parameters(), 
                          lr=config['lr'], 
                          weight_decay=config['weight_decay'])
    scaler = GradScaler()

    # Load datasets and create dataloaders
    channel_groups = load_data(config['channel_groups_path'])
    dataset = SetTransformerDataset(config, channel_groups, split='pretrain')
    dataset_val = SetTransformerDataset(config, channel_groups, split='validation')

    sampler = DistributedSampler(dataset)
    sampler_val = DistributedSampler(dataset_val)
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        sampler=sampler,
        num_workers=config['num_workers'],
        collate_fn=collate_fn_MAE,
        drop_last=True
    )
    
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=config['batch_size'],
        sampler=sampler_val,
        num_workers=int(config['num_workers'] // 4),
        collate_fn=collate_fn_MAE
    )

    # Create model save directory
    if rank == 0:
        model_save_dir = os.path.join(config['save_path'], config['model'])
        os.makedirs(model_save_dir, exist_ok=True)
    
    # Load checkpoint if provided
    if config['checkpoint_path'] is not None:
        checkpoint = torch.load(config['checkpoint_path'], map_location=device)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
    
    if rank == 0:
        print(f'number of iterations per epoch: {len(dataloader)}')

    # Training loop
    for epoch in range(config['epochs']):
        sampler.set_epoch(epoch)
        model.train()
        
        # Initialize loss accumulators for the epoch
        accumulated_losses = {
            'amp': 0.0, 'phase': 0.0,
            'amp_bas': 0.0, 'phase_bas': 0.0,
            'amp_resp': 0.0, 'phase_resp': 0.0,
            'amp_ekg': 0.0, 'phase_ekg': 0.0,
            'amp_emg': 0.0, 'phase_emg': 0.0,
            'amp_masked': 0.0, 'phase_masked': 0.0,
            'amp_bas_masked': 0.0, 'phase_bas_masked': 0.0,
            'amp_resp_masked': 0.0, 'phase_resp_masked': 0.0,
            'amp_ekg_masked': 0.0, 'phase_ekg_masked': 0.0,
            'amp_emg_masked': 0.0, 'phase_emg_masked': 0.0,
            'amp_non_masked': 0.0, 'phase_non_masked': 0.0,
            'amp_bas_non_masked': 0.0, 'phase_bas_non_masked': 0.0,
            'amp_resp_non_masked': 0.0, 'phase_resp_non_masked': 0.0,
            'amp_ekg_non_masked': 0.0, 'phase_ekg_non_masked': 0.0,
            'amp_emg_non_masked': 0.0, 'phase_emg_non_masked': 0.0
        }
        
        epoch_losses = {k: 0.0 for k in accumulated_losses.keys()}

        if rank == 0:
            loop = tqdm(enumerate(dataloader), 
                       total=len(dataloader), 
                       desc=f"Epoch [{epoch+1}/{config['epochs']}]")
        else:
            loop = enumerate(dataloader)

        optimizer.zero_grad()

        for batch_idx, batch in loop:
            # Process batch data
            padded_batch_list, mask_list, file_paths, dset_names_list, chunk_starts = batch
            padded_batch_list = padded_batch_list.to(device, non_blocking=True)

            # Forward pass with mixed precision
            with autocast():
                output, embedding, mask = model(padded_batch_list)
                losses = get_losses_fft(padded_batch_list, output, criterion, mask)
                
                # Combine amplitude and phase losses for optimization
                total_loss = (losses[12] + losses[13]) / 2  # loss_amp_masked + loss_phase_masked
                scaled_loss = total_loss / config['accumulation_interval']

            # Backward pass with gradient scaling
            scaler.scale(scaled_loss).backward()
            
            # Accumulate all losses
            for idx, (key, loss_idx) in enumerate([
                ('amp', 2), ('phase', 3),
                ('amp_bas', 4), ('phase_bas', 5),
                ('amp_resp', 6), ('phase_resp', 7),
                ('amp_ekg', 8), ('phase_ekg', 9),
                ('amp_emg', 10), ('phase_emg', 11),
                ('amp_masked', 12), ('phase_masked', 13),
                ('amp_bas_masked', 14), ('phase_bas_masked', 15),
                ('amp_resp_masked', 16), ('phase_resp_masked', 17),
                ('amp_ekg_masked', 18), ('phase_ekg_masked', 19),
                ('amp_emg_masked', 20), ('phase_emg_masked', 21),
                ('amp_non_masked', 22), ('phase_non_masked', 23),
                ('amp_bas_non_masked', 24), ('phase_bas_non_masked', 25),
                ('amp_resp_non_masked', 26), ('phase_resp_non_masked', 27),
                ('amp_ekg_non_masked', 28), ('phase_ekg_non_masked', 29),
                ('amp_emg_non_masked', 30), ('phase_emg_non_masked', 31)
            ]):
                accumulated_losses[key] += losses[loss_idx].item()
                epoch_losses[key] += losses[loss_idx].item()

            # Determine if we should update the model
            is_accumulation_step = (batch_idx + 1) % config['accumulation_interval'] == 0
            is_last_batch = (batch_idx + 1) == len(dataloader)
            
            if is_accumulation_step or is_last_batch:

                # Gradient clipping
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 2.0)

                # Calculate average losses
                steps_in_interval = batch_idx % config['accumulation_interval'] + 1
                accumulations = min(config['accumulation_interval'], steps_in_interval)
                avg_losses = {k: v / accumulations for k, v in accumulated_losses.items()}
                
                # Log progress
                if rank == 0:
                    loop.set_postfix(loss=(avg_losses['amp'] + avg_losses['phase'])/2)
                    if config['use_wandb']:
                        wandb_logs = {
                            'total_loss': (avg_losses['amp'] + avg_losses['phase'])/2,
                            'learning_rate': optimizer.param_groups[0]['lr'],
                            'scale': scaler.get_scale(),
                            'grad_norm': grad_norm.item(),  # Log the gradient norm
                            'epoch': epoch,
                            'batch': batch_idx
                        }
                        # Add all individual losses to wandb logs
                        for key, value in avg_losses.items():
                            wandb_logs[f'MSE_{key}'] = value
                        wandb.log(wandb_logs)

                # Update model
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Save checkpoint if needed
                if rank == 0 and (batch_idx + 1) % config['save_iter'] == 0:
                    model_save_path = os.path.join(
                        model_save_dir,
                        f'{current_timestamp}_epoch_{epoch}_batch_{batch_idx}.pt'
                    )
                    torch.save({
                        'epoch': epoch,
                        'batch': batch_idx,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'loss': (avg_losses['amp'] + avg_losses['phase'])/2,
                    }, model_save_path)

                # Reset accumulators
                for key in accumulated_losses:
                    accumulated_losses[key] = 0.0

            # Validation loop
            if (batch_idx + 1) % config['eval_iter'] == 0:
                print(f"Evaluating model at epoch {epoch} and batch {batch_idx}") if rank == 0 else None
                model.eval()
                sampler_val.set_epoch(epoch)

                local_loss_accumulators = {
                    'amp_val': 0.0, 'phase_val': 0.0,
                    'amp_bas_val': 0.0, 'phase_bas_val': 0.0,
                    'amp_resp_val': 0.0, 'phase_resp_val': 0.0,
                    'amp_ekg_val': 0.0, 'phase_ekg_val': 0.0,
                    'amp_emg_val': 0.0, 'phase_emg_val': 0.0,
                    'amp_masked_val': 0.0, 'phase_masked_val': 0.0,
                    'amp_bas_masked_val': 0.0, 'phase_bas_masked_val': 0.0,
                    'amp_resp_masked_val': 0.0, 'phase_resp_masked_val': 0.0,
                    'amp_ekg_masked_val': 0.0, 'phase_ekg_masked_val': 0.0,
                    'amp_emg_masked_val': 0.0, 'phase_emg_masked_val': 0.0,
                    'amp_non_masked_val': 0.0, 'phase_non_masked_val': 0.0,
                    'amp_bas_non_masked_val': 0.0, 'phase_bas_non_masked_val': 0.0,
                    'amp_resp_non_masked_val': 0.0, 'phase_resp_non_masked_val': 0.0,
                    'amp_ekg_non_masked_val': 0.0, 'phase_ekg_non_masked_val': 0.0,
                    'amp_emg_non_masked_val': 0.0, 'phase_emg_non_masked_val': 0.0
                }

                with torch.no_grad():
                    for val_batch in tqdm(dataloader_val, 
                                        desc="Validation loop",
                                        disable=rank != 0):
                        padded_batch_list_val, mask_list_val, file_paths_val, dset_names_list_val, chunk_starts_val = val_batch
                        padded_batch_list_val = padded_batch_list_val.to(device, non_blocking=True)
                        
                        output_val, embedding_val, mask_val = model(padded_batch_list_val)
                        val_losses = get_losses_fft(padded_batch_list_val, output_val, criterion, mask_val)
                        
                        # Accumulate validation losses
                        for idx, (key, loss_idx) in enumerate([
                            ('amp_val', 2), ('phase_val', 3),
                            ('amp_bas_val', 4), ('phase_bas_val', 5),
                            ('amp_resp_val', 6), ('phase_resp_val', 7),
                            ('amp_ekg_val', 8), ('phase_ekg_val', 9),
                            ('amp_emg_val', 10), ('phase_emg_val', 11),
                            ('amp_masked_val', 12), ('phase_masked_val', 13),
                            ('amp_bas_masked_val', 14), ('phase_bas_masked_val', 15),
                            ('amp_resp_masked_val', 16), ('phase_resp_masked_val', 17),
                            ('amp_ekg_masked_val', 18), ('phase_ekg_masked_val', 19),
                            ('amp_emg_masked_val', 20), ('phase_emg_masked_val', 21),
                            ('amp_non_masked_val', 22), ('phase_non_masked_val', 23),
                            ('amp_bas_non_masked_val', 24), ('phase_bas_non_masked_val', 25),
                            ('amp_resp_non_masked_val', 26), ('phase_resp_non_masked_val', 27),
                            ('amp_ekg_non_masked_val', 28), ('phase_ekg_non_masked_val', 29),
                            ('amp_emg_non_masked_val', 30), ('phase_emg_non_masked_val', 31)
                        ]):
                            local_loss_accumulators[key] += val_losses[loss_idx].item()

                    # Synchronize losses across all processes using distributed computing
                    world_size = float(dist.get_world_size())
                    avg_losses_val = {}
                    
                    # Calculate average losses across all processes
                    for key, local_val in local_loss_accumulators.items():
                        # Convert local loss to tensor for all_reduce operation
                        loss_tensor = torch.tensor([local_val], device=device)
                        # Sum up losses from all processes
                        dist.all_reduce(loss_tensor)
                        # Calculate the average loss across all processes and batches
                        avg_losses_val[key] = (loss_tensor.item() / world_size) / len(dataloader_val)

                    # Only rank 0 handles logging and early stopping
                    stop_signal = torch.tensor(0, device=device)
                    if rank == 0:
                        # Calculate combined validation loss for monitoring
                        total_val_loss = (avg_losses_val['amp_masked_val'] + avg_losses_val['phase_masked_val']) / 2
                        print(f"Validation Loss: {total_val_loss:.4f}")
                        
                        # Log validation metrics if wandb is enabled
                        if config['use_wandb']:
                            wandb_logs_val = {
                                'total_val_loss': total_val_loss,
                                'epoch': epoch,
                                'batch': batch_idx
                            }
                            # Add all individual validation losses to wandb logs
                            for key, value in avg_losses_val.items():
                                wandb_logs_val[f'MSE_{key}'] = value
                            wandb.log(wandb_logs_val)

                        # Handle model saving and early stopping
                        if total_val_loss < min_avg_loss_val:
                            min_avg_loss_val = total_val_loss
                            print(f"New best validation loss: {min_avg_loss_val:.4f}")
                            
                            # Save the best model checkpoint
                            model_save_path_best_val = os.path.join(
                                model_save_dir,
                                f'{current_timestamp}_epoch_{epoch}_batch_{batch_idx}_val_loss_{min_avg_loss_val:.4f}.pt'
                            )
                            torch.save({
                                'epoch': epoch,
                                'batch': batch_idx,
                                'model_state_dict': model.module.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scaler_state_dict': scaler.state_dict(),
                                'loss': total_val_loss,
                                'min_val_loss': min_avg_loss_val
                            }, model_save_path_best_val)
                            
                            # Reset patience since we found a better model
                            patience = config['patience']
                        else:
                            # Decrease patience if validation loss didn't improve
                            patience -= 1
                            if patience == 0:
                                print(f"Early stopping triggered after {config['patience']} evaluations without improvement")
                                if config['use_wandb']:
                                    wandb.finish()
                                stop_signal = torch.tensor(1, device=device)
                            else:
                                stop_signal = torch.tensor(0, device=device)
                    else:
                        # Non-rank 0 processes initialize stop_signal
                        stop_signal = torch.tensor(0, device=device)

                    # Synchronize processes before and after broadcasting stop signal
                    dist.barrier()
                    dist.broadcast(stop_signal, src=0)
                    dist.barrier()

                    # If stop signal is received, clean up and exit
                    if stop_signal.item() == 1:
                        dist.destroy_process_group()
                        return

                    # Return to training mode after validation
                    model.train()

        # End of epoch processing
        if rank == 0:
            # Calculate and log epoch-level statistics
            avg_epoch_losses = {k: v / len(dataloader) for k, v in epoch_losses.items()}
            total_epoch_loss = (avg_epoch_losses['amp'] + avg_epoch_losses['phase']) / 2
            
            print(f"Epoch [{epoch+1}/{config['epochs']}] Average Loss: {total_epoch_loss:.4f}")
            
            if config['use_wandb']:
                wandb_logs_epoch = {
                    'epoch_total_loss': total_epoch_loss,
                    'epoch': epoch,
                }
                for key, value in avg_epoch_losses.items():
                    wandb_logs_epoch[f'epoch_MSE_{key}'] = value
                wandb.log(wandb_logs_epoch)
            
            # Save epoch checkpoint
            model_save_path = os.path.join(
                model_save_dir,
                f'{current_timestamp}_epoch_{epoch}.pt'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': total_epoch_loss,
                'min_val_loss': min_avg_loss_val
            }, model_save_path)

    # Clean up at the end of training
    dist.destroy_process_group()