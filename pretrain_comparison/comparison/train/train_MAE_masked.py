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
from utils import load_data, get_losses, initialize_mae_weights
import pickle
import yaml

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

    # Initialize model
    model = MAE(mask_ratio = config['mask_ratio'], input_size=640, output_size=128)
    model.to(device)
    initialize_mae_weights(model)

    # Wrap model with DDP
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    model = DDP(model, 
                device_ids=[local_rank], 
                output_device=local_rank, 
                find_unused_parameters=True)

    # Define loss function and optimizer
    criterion = nn.MSELoss(reduction='none').to(device)
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

    # Create model save directory
    if rank == 0:
        model_save_dir = os.path.join(config['save_path'], config['model'])
        os.makedirs(model_save_dir, exist_ok=True)
    
    if config['checkpoint_path'] is not None:
        checkpoint = torch.load(config['checkpoint_path'], map_location=device)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
    
    if rank == 0:
        print(f'number of iterations per epoch: {len(dataloader)}')

    # Training loop
    num_epochs = config['epochs']
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        epoch_loss_bas = 0.0
        epoch_loss_resp = 0.0
        epoch_loss_ekg = 0.0
        epoch_loss_emg = 0.0
        
        accumulated_loss = 0.0
        accumulated_loss_bas = 0.0
        accumulated_loss_resp = 0.0
        accumulated_loss_ekg = 0.0
        accumulated_loss_emg = 0.0

        accumulated_loss_masked = 0.0
        accumulated_loss_bas_masked = 0.0
        accumulated_loss_resp_masked = 0.0
        accumulated_loss_ekg_masked = 0.0
        accumulated_loss_emg_masked = 0.0

        accumulated_loss_non_masked = 0.0
        accumulated_loss_bas_non_masked = 0.0
        accumulated_loss_resp_non_masked = 0.0
        accumulated_loss_ekg_non_masked = 0.0
        accumulated_loss_emg_non_masked = 0.0


        if rank == 0:
            loop = tqdm(enumerate(dataloader), 
                       total=len(dataloader), 
                       desc=f"Epoch [{epoch+1}/{num_epochs}]")
        else:
            loop = enumerate(dataloader)

        optimizer.zero_grad()  # Zero gradients at the start of epoch
        
        for batch_idx, batch in loop:
            # Unpack and move data to device
            padded_batch_list, mask_list, file_paths, dset_names_list, chunk_starts = batch
            padded_batch_list = padded_batch_list.to(device, non_blocking=True)

            # Forward pass with mixed precision
            with autocast():
                output, embedding, mask = model(padded_batch_list)
                
                loss_full, loss, loss_bas, loss_resp, loss_ekg, loss_emg, loss_masked, loss_bas_masked, loss_resp_masked, loss_ekg_masked, loss_emg_masked, loss_non_masked, loss_bas_non_masked, loss_resp_non_masked, loss_ekg_non_masked, loss_emg_non_masked = get_losses(padded_batch_list, output, criterion, mask)
                #get fourier losses here with 
                loss = loss_masked

                # Scale loss by accumulation interval for proper gradient accumulation
                scaled_loss = loss / config['accumulation_interval']
            
            # Backward pass with gradient scaling
            scaler.scale(scaled_loss).backward()
            
            # Accumulate the true loss (not scaled) for logging
            accumulated_loss += loss.item()
            accumulated_loss_bas += loss_bas.item()
            accumulated_loss_resp += loss_resp.item()
            accumulated_loss_ekg += loss_ekg.item()
            accumulated_loss_emg += loss_emg.item()

            accumulated_loss_masked += loss_masked.item()
            accumulated_loss_bas_masked += loss_bas_masked.item()
            accumulated_loss_resp_masked += loss_resp_masked.item()
            accumulated_loss_ekg_masked += loss_ekg_masked.item()
            accumulated_loss_emg_masked += loss_emg_masked.item()

            accumulated_loss_non_masked += loss_non_masked.item()
            accumulated_loss_bas_non_masked += loss_bas_non_masked.item()
            accumulated_loss_resp_non_masked += loss_resp_non_masked.item()
            accumulated_loss_ekg_non_masked += loss_ekg_non_masked.item()
            accumulated_loss_emg_non_masked += loss_emg_non_masked.item()

            
            # Determine if we should step the optimizer
            is_accumulation_step = (batch_idx + 1) % config['accumulation_interval'] == 0
            is_last_batch = (batch_idx + 1) == len(dataloader)
            
            if is_accumulation_step or is_last_batch:
                # Calculate average loss over accumulation interval
                steps_in_interval = batch_idx % config['accumulation_interval'] + 1
                acummulations = min(config['accumulation_interval'], steps_in_interval)
                avg_accumulated_loss = accumulated_loss / acummulations
                avg_accumulated_loss_bas = accumulated_loss_bas / acummulations
                avg_accumulated_loss_resp = accumulated_loss_resp / acummulations
                avg_accumulated_loss_ekg = accumulated_loss_ekg / acummulations
                avg_accumulated_loss_emg = accumulated_loss_emg / acummulations

                avg_accumulated_loss_masked = accumulated_loss_masked / acummulations
                avg_accumulated_loss_bas_masked = accumulated_loss_bas_masked / acummulations
                avg_accumulated_loss_resp_masked = accumulated_loss_resp_masked / acummulations
                avg_accumulated_loss_ekg_masked = accumulated_loss_ekg_masked / acummulations
                avg_accumulated_loss_emg_masked = accumulated_loss_emg_masked / acummulations

                avg_accumulated_loss_non_masked = accumulated_loss_non_masked / acummulations
                avg_accumulated_loss_bas_non_masked = accumulated_loss_bas_non_masked / acummulations
                avg_accumulated_loss_resp_non_masked = accumulated_loss_resp_non_masked / acummulations
                avg_accumulated_loss_ekg_non_masked = accumulated_loss_ekg_non_masked / acummulations
                avg_accumulated_loss_emg_non_masked = accumulated_loss_emg_non_masked / acummulations

                
                # Log the actual accumulated loss before stepping
                if rank == 0:
                    loop.set_postfix(loss=avg_accumulated_loss)
                    if config['use_wandb']:
                        #also log the loss for each of the modality types and with/without masking
                        wandb.log({
                            'MSE': avg_accumulated_loss,
                            'MSE_bas': avg_accumulated_loss_bas,
                            'MSE_resp': avg_accumulated_loss_resp,
                            'MSE_ekg': avg_accumulated_loss_ekg,
                            'MSE_emg': avg_accumulated_loss_emg,
                            'MSE_masked': avg_accumulated_loss_masked,
                            'MSE_bas_masked': avg_accumulated_loss_bas_masked,
                            'MSE_resp_masked': avg_accumulated_loss_resp_masked,
                            'MSE_ekg_masked': avg_accumulated_loss_ekg_masked,
                            'MSE_emg_masked': avg_accumulated_loss_emg_masked,
                            'MSE_non_masked': avg_accumulated_loss_non_masked,
                            'MSE_bas_non_masked': avg_accumulated_loss_bas_non_masked,
                            'MSE_resp_non_masked': avg_accumulated_loss_resp_non_masked,
                            'MSE_ekg_non_masked': avg_accumulated_loss_ekg_non_masked,
                            'MSE_emg_non_masked': avg_accumulated_loss_emg_non_masked,
                            'epoch': epoch,
                            'batch': batch_idx,
                            'learning_rate': optimizer.param_groups[0]['lr'],
                            'scale': scaler.get_scale()  # Log the current scale factor
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

                    # Initialize loss accumulators
                    local_loss_accumulators = {
                        'val': 0.0, 'bas_val': 0.0, 'resp_val': 0.0, 'ekg_val': 0.0, 'emg_val': 0.0,
                        'masked_val': 0.0, 'bas_masked_val': 0.0, 'resp_masked_val': 0.0, 'ekg_masked_val': 0.0, 'emg_masked_val': 0.0,
                        'non_masked_val': 0.0, 'bas_non_masked_val': 0.0, 'resp_non_masked_val': 0.0, 'ekg_non_masked_val': 0.0, 'emg_non_masked_val': 0.0
                    }

                    with torch.no_grad():
                        for val_idx, batch_val in tqdm(enumerate(dataloader_val), 
                                                    total=len(dataloader_val), 
                                                    desc=f"Validation loop",
                                                    disable=rank != 0):
                            padded_batch_list_val, mask_list_val, file_paths_val, dset_names_list_val, chunk_starts_val = batch_val
                            padded_batch_list_val = padded_batch_list_val.to(device, non_blocking=True)
                            output_val, embedding_val, mask_val = model(padded_batch_list_val)
                            
                            losses = get_losses(padded_batch_list_val, output_val, criterion, mask_val)
                            
                            # Accumulate all losses
                            local_loss_accumulators['val'] += losses[1].item()  # loss_val
                            local_loss_accumulators['bas_val'] += losses[2].item()  # loss_bas_val
                            local_loss_accumulators['resp_val'] += losses[3].item()  # loss_resp_val
                            local_loss_accumulators['ekg_val'] += losses[4].item()  # loss_ekg_val
                            local_loss_accumulators['emg_val'] += losses[5].item()  # loss_emg_val
                            local_loss_accumulators['masked_val'] += losses[6].item()  # loss_masked_val
                            local_loss_accumulators['bas_masked_val'] += losses[7].item()
                            local_loss_accumulators['resp_masked_val'] += losses[8].item()
                            local_loss_accumulators['ekg_masked_val'] += losses[9].item()
                            local_loss_accumulators['emg_masked_val'] += losses[10].item()
                            local_loss_accumulators['non_masked_val'] += losses[11].item()
                            local_loss_accumulators['bas_non_masked_val'] += losses[12].item()
                            local_loss_accumulators['resp_non_masked_val'] += losses[13].item()
                            local_loss_accumulators['ekg_non_masked_val'] += losses[14].item()
                            local_loss_accumulators['emg_non_masked_val'] += losses[15].item()

                        # Synchronize all losses across processes
                        world_size = float(dist.get_world_size())
                        avg_losses = {}
                        for key, local_val in local_loss_accumulators.items():
                            loss_tensor = torch.tensor([local_val], device=device)
                            dist.all_reduce(loss_tensor)
                            avg_losses[key] = (loss_tensor.item() / world_size) / len(dataloader_val)

                        # Only rank 0 handles logging and early stopping
                        if rank == 0:
                            print(f"Validation Loss: {avg_losses['val']:.4f}")
                            
                            if config['use_wandb']:
                                wandb.log({
                                    'MSE_val': avg_losses['val'],
                                    'MSE_bas_val': avg_losses['bas_val'],
                                    'MSE_resp_val': avg_losses['resp_val'],
                                    'MSE_ekg_val': avg_losses['ekg_val'],
                                    'MSE_emg_val': avg_losses['emg_val'],
                                    'MSE_masked_val': avg_losses['masked_val'],
                                    'MSE_bas_masked_val': avg_losses['bas_masked_val'],
                                    'MSE_resp_masked_val': avg_losses['resp_masked_val'],
                                    'MSE_ekg_masked_val': avg_losses['ekg_masked_val'],
                                    'MSE_emg_masked_val': avg_losses['emg_masked_val'],
                                    'MSE_non_masked_val': avg_losses['non_masked_val'],
                                    'MSE_bas_non_masked_val': avg_losses['bas_non_masked_val'],
                                    'MSE_resp_non_masked_val': avg_losses['resp_non_masked_val'],
                                    'MSE_ekg_non_masked_val': avg_losses['ekg_non_masked_val'],
                                    'MSE_emg_non_masked_val': avg_losses['emg_non_masked_val'],
                                    'epoch': epoch,
                                    'batch': batch_idx
                                })

                        # Handle early stopping
                        stop_signal = torch.tensor(0, device=device)
                        if rank == 0:
                            if min_avg_loss_val > avg_losses['val']:
                                min_avg_loss_val = avg_losses['val']
                                print(f"Saving model with validation loss {min_avg_loss_val}")
                                model_save_path_best_val = os.path.join(
                                    model_save_dir,
                                    f'{current_timestamp}_epoch_{epoch}_batch_{batch_idx}_val_loss_{min_avg_loss_val}.pt'
                                )
                                torch.save({
                                    'epoch': epoch,
                                    'batch': batch_idx,
                                    'model_state_dict': model.module.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'scaler_state_dict': scaler.state_dict(),
                                    'loss': avg_accumulated_loss,
                                }, model_save_path_best_val)
                                patience = config['patience']
                            else:
                                patience -= 1
                                if patience == 0:
                                    print(f"Early stopping triggered")
                                    if config['use_wandb']:
                                        wandb.finish()
                                    stop_signal.fill_(1)

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
                accumulated_loss_bas = 0.0
                accumulated_loss_resp = 0.0
                accumulated_loss_ekg = 0.0
                accumulated_loss_emg = 0.0

                accumulated_loss_masked = 0.0
                accumulated_loss_bas_masked = 0.0
                accumulated_loss_resp_masked = 0.0
                accumulated_loss_ekg_masked = 0.0
                accumulated_loss_emg_masked = 0.0

                accumulated_loss_non_masked = 0.0
                accumulated_loss_bas_non_masked = 0.0
                accumulated_loss_resp_non_masked = 0.0
                accumulated_loss_ekg_non_masked = 0.0
                accumulated_loss_emg_non_masked = 0.0

                

            
            epoch_loss += loss.item()
            epoch_loss_bas += loss_bas.item()
            epoch_loss_resp += loss_resp.item()
            epoch_loss_ekg += loss_ekg.item()
            epoch_loss_emg += loss_emg.item()

        # Calculate and log epoch statistics
        if rank == 0:
            avg_epoch_loss = epoch_loss / len(dataloader)
            avg_epoch_loss_bas = epoch_loss_bas / len(dataloader)
            avg_epoch_loss_resp = epoch_loss_resp / len(dataloader)
            avg_epoch_loss_ekg = epoch_loss_ekg / len(dataloader)
            avg_epoch_loss_emg = epoch_loss_emg / len(dataloader)
            print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_epoch_loss:.4f}")
            
            if config['use_wandb']:
                wandb.log({
                    'epoch_loss': avg_epoch_loss,
                    'epoch_loss_bas': avg_epoch_loss_bas,
                    'epoch_loss_resp': avg_epoch_loss_resp,
                    'epoch_loss_ekg': avg_epoch_loss_ekg,
                    'epoch_loss_emg': avg_epoch_loss_emg,
                    'epoch': epoch,
                })
            
            # Save epoch checkpoint
            model_save_path = os.path.join(
                model_save_dir,
                f'{current_timestamp}_epoch_{epoch}.pt'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),  # Save scaler state
                'loss': avg_epoch_loss,
            }, model_save_path)

    # Cleanup
    dist.destroy_process_group()