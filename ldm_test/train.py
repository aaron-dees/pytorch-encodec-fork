#### RAVE-Latent Diffusion
#### https://github.com/moiseshorta/RAVE-Latent-Diffusion
####
#### Author: Moisés Horta Valenzuela / @hexorcismos
#### Year: 2023
import sys
sys.path.append("..")

import torch.nn as nn
import torchaudio
import hydra

from model import EncodecModel
import customAudioDataset as data
from customAudioDataset import collate_fn
from utilities import export_latents, reconstruct_encodec_frames

import argparse
import torch.multiprocessing as mp
import torch
import os
import time
import datetime
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision


from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps:0")
else:
    device = torch.device("cpu")
current_date = datetime.date.today()
# device='cpu'


class RaveDataset(Dataset):
    def __init__(self, latent_folder, latent_files):
        self.latent_folder = latent_folder
        self.latent_files = latent_files
        self.latent_data = []

        for latent_file in self.latent_files:
            latent_path = os.path.join(self.latent_folder, latent_file)
            z = np.load(latent_path)
            z = torch.from_numpy(z).float().squeeze()
            self.latent_data.append(z)

        self.latent_size = self.latent_data[0].shape[0]

    def __len__(self):
        return len(self.latent_data)

    def __getitem__(self, index):
        return self.latent_data[index]

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with a new dataset.")
    parser.add_argument("--name", type=str, default=f"run_{current_date}", help="Name of your training run.")
    parser.add_argument("--latent_folder", type=str, default="./latents/", help="Path to the directory containing the latent files.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Resume training from checkpoint.")
    parser.add_argument("--save_out_path", type=str, default="./runs/", help="Path to the directory where the model checkpoints will be saved.")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Ratio for splitting the dataset into training and validation sets.")
    parser.add_argument("--max_epochs", type=int, default=25000, help="Maximum epochs to train model.")
    parser.add_argument("--scheduler_steps", type=int, default=100, help="Diffusion steps for scheduler.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--accumulation_steps", type=int, default=2, help="Number of gradient accumulation steps.")
    parser.add_argument("--save_interval", type=int, default=50, help="Interval (number of epochs) at which to save the model.")
    parser.add_argument("--finetune", type=bool, default=False, help="Finetune model.")
    return parser.parse_args()

def set_seed(seed=664):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def resume_from_checkpoint(checkpoint_path, model, optimizer, scheduler):
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        if 'model_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint and 'scheduler_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
        else:
            print("The checkpoint file does not contain the required keys. Training will start from scratch.")
            start_epoch = 0
    else:
        start_epoch = 0

    return start_epoch

@hydra.main(config_path='./config', config_name='config')
def main(config):


    writer = SummaryWriter(log_dir="./runs/diffusion_run")


    checkpoint_path = None

    global best_loss
    global best_epoch
    best_epoch = None
    best_loss = float('inf')

    os.makedirs(config.diffusion.save_out_path, exist_ok=True)

    # latent_files = [f for f in os.listdir(latent_folder) if f.endswith(".npy")]

    # random.shuffle(latent_files)
    # split_index = int(len(latent_files) * split_ratio)
    # train_latent_files = latent_files[:split_index]
    # val_latent_files = latent_files[split_index:]

    # train_dataset = RaveDataset(latent_folder, train_latent_files)
    # val_dataset = RaveDataset(latent_folder, val_latent_files)

    # rave_dims = train_dataset.latent_size

    # batch_size = args.batch_size

    # train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    # val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    set_seed(664)
    
    trainset = data.CustomAudioDataset(config=config)
    testset = data.CustomAudioDataset(config=config,mode='test')

    train_sampler = None
    test_sampler = None


    wav_trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=config.datasets.batch_size,
        sampler=train_sampler, 
        shuffle=(train_sampler is None), collate_fn=collate_fn,
        pin_memory=config.datasets.pin_memory)

    wav_testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=config.datasets.batch_size,
        sampler=test_sampler, 
        shuffle=False, collate_fn=collate_fn,
        pin_memory=config.datasets.pin_memory)
    print(f"There are {len(wav_trainloader)} data to train the EnCodec")
    print(f"There are {len(wav_testloader)} data to test the EnCodec")

    w_model = EncodecModel._get_model(
        config.model.target_bandwidths, 
        config.model.sample_rate, 
        config.model.channels,
        causal=config.model.causal, model_norm=config.model.norm, 
        audio_normalize=config.model.audio_normalize,
        segment=config.model.segment, name=config.model.name,
        ratios=config.model.ratios,
    )

    w_model_checkpoint = torch.load(config.diffusion.waveform_model_path, map_location='cpu')
    w_model.load_state_dict(w_model_checkpoint['model_state_dict'])

    w_model.to(device)
    w_model.eval()

    print("model loaded")

    print("Exporting Latents...")

    train_latents, test_latents = export_latents(w_model, wav_trainloader, wav_testloader, config.datasets.batch_size, DEVICE)
    VAE_DIMS = train_latents.shape[1]

    train_data_loader = torch.utils.data.DataLoader(
        train_latents,
        shuffle=True,
        batch_size=config.datasets.batch_size,
        # num_workers=0,
        # pin_memory=False,
    )
    val_data_loader = torch.utils.data.DataLoader(
        test_latents,
        shuffle=True,
        batch_size=config.datasets.batch_size,
        # num_workers=0,
        # pin_memory=False,
    )

    # Dynamically choose a UNet depth that fits the exported latent temporal length.
    # Ensure the product of downsampling factors <= sample temporal length (e.g. 256).
    sample_len = train_latents[0].shape[-1]
    print(f"Sample latent temporal length = {sample_len}")

    # base architecture lists (original)
    base_channels = [256, 256, 256, 256, 512, 512, 512, 768, 768]
    base_factors  = [1,   4,   4,   4,   2,   2,   2,   2,   2]
    base_items    = [1,   2,   2,   2,   2,   2,   2,   4,   4]
    base_attns    = [0,   0,   0,   0,   0,   1,   1,   1,   1]

    prod = 1
    max_idx = 0
    for i, f in enumerate(base_factors):
        prod *= f
        if prod <= sample_len:
            max_idx = i + 1
        else:
            break
    max_idx = max(1, max_idx)

    channels = base_channels[:max_idx]
    factors  = base_factors[:max_idx]
    items    = base_items[:max_idx]
    attentions = base_attns[:max_idx]

    print("Using UNet config:")
    print("  depth =", max_idx)
    print("  channels =", channels)
    print("  factors  =", factors, " (product =", int(np.prod(factors)), ")")
    print("  items    =", items)
    print("  attentions=", attentions)

    model = DiffusionModel(
        net_t=UNetV0,
        in_channels=VAE_DIMS,
        channels=channels,
        factors=factors,
        items=items,
        attentions=attentions,
        attention_heads=12,
        attention_features=64,
        diffusion_t=VDiffusion,
        sampler_t=VSampler,
    ).to(device)

    print("Model Architecture:")
    # print(model)
    print("\nModel Parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")
    print(f"Number of trainable parameters: {trainable_params}\n")

    # print("Training:", len(train_latent_files))
    # print("Validation:", len(val_latent_files))

    if checkpoint_path != None:
        print(f"Resuming training from: {checkpoint_path}\n")

    if not config.diffusion.finetune:
        ##### TRAIN FROM SCRATCH
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.diffusion.scheduler_steps, gamma=0.99)
        start_epoch = resume_from_checkpoint(checkpoint_path, model, optimizer, scheduler)
    else:
        #### FINETUNE
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5) # Change the learning rate
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.diffusion.scheduler_steps, eta_min=1e-6) # Replace the StepLR scheduler with the CosineAnnealingLR scheduler
        start_epoch = resume_from_checkpoint(checkpoint_path, model, optimizer, scheduler)

    accumulation_steps = config.diffusion.accumulation_steps 

    for i in range(start_epoch, config.diffusion.epochs):
        model.train()
        train_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_data_loader):
            # print(len(batch))
            batch_rave_tensor = batch.to(device)
            print("Batch Size: ", batch_rave_tensor.shape)

            loss = model(batch_rave_tensor)
            print(loss.shape)

            train_loss += loss.item()

            if (step + 1) % accumulation_steps == 0:
                loss = loss / accumulation_steps
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        train_loss /= len(train_data_loader)
        print(f"Epoch {i+1}, train loss: {train_loss}")
        writer.add_scalar("Loss/Train", train_loss, i+1)

        # random.shuffle(train_dataset.latent_files)

        with torch.no_grad():
            model.eval()

            val_loss = 0
            for batch in val_data_loader:
                batch_rave_tensor = batch.to(device)

                loss = model(batch_rave_tensor)

                val_loss += loss.item()

            val_loss /= len(val_data_loader)
            print(f"Epoch {i+1}, validation loss: {val_loss}")
            writer.add_scalar("Loss/Val", val_loss, i+1)

            # LATENT ANALYSIS DOESNT REALLY WORK WELL AS IT IS NOT NECISARILY TRYING TO MATCH AND EXACT TRJECTORY FROM THE NOISE.
            # BETTER TEST TO RESYNTHESIZE THE SOUND.
            if i % config.diffusion.view_analysis_interval == 0:
                noise = torch.randn(1, VAE_DIMS, sample_len).to(device)
                noise = noise * config.diffusion.temperature
                diff = model.sample(noise[:,:,:256], num_steps=config.diffusion.scheduler_steps, show_progress=True)

                frames = diff.permute(0,2,1)
                # frames = diff
                batch_rave_tensor = batch_rave_tensor.permute(0,2,1)
                # print(frames)

                print("Reconstructing Audio")
                # print(batch_rave_tensor.shape)
                pred_encodec_frames = reconstruct_encodec_frames(diff)
                target_encodec_frames = reconstruct_encodec_frames(batch_rave_tensor.permute(0,2,1))
                pred_audio = w_model.decode(pred_encodec_frames)
                target_audio = w_model.decode(target_encodec_frames)
                torchaudio.save(f'./recon.wav', pred_audio[0].cpu(), w_model.sample_rate, channels_first=True)
                torchaudio.save(f'./orig.wav', target_audio[0].cpu(), w_model.sample_rate, channels_first=True)

                writer.add_audio("Recon/GenLatent", pred_audio, sample_rate=w_model.sample_rate, global_step=i+1)
                writer.add_audio("Recon/OrigLatent", target_audio, sample_rate=w_model.sample_rate, global_step=i+1)

                import matplotlib.pyplot as plt
                from sklearn.decomposition import PCA
                # Use first batch of validation for plotting
                target_seq = batch_rave_tensor[0].cpu().numpy()
                pred_seq = frames[0].cpu().numpy()
                all_seq = np.vstack([target_seq, pred_seq])
                pca = PCA(n_components=3)
                Zp = pca.fit_transform(all_seq)
                Zp_target = Zp[:target_seq.shape[0]]
                Zp_pred = Zp[pred_seq.shape[0]:]
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                plt.plot(Zp_target[:, 0], Zp_target[:, 1], '-o', markersize=3, alpha=0.6, label='Target')
                plt.plot(Zp_pred[:, 0], Zp_pred[:, 1], '-o', markersize=3, alpha=0.6, label='Prediction')
                plt.xlabel('PC1')
                plt.ylabel('PC2')
                plt.title('Latent Trajectories (Top 2 PCA)')
                plt.legend()
                plt.grid(True)
                from mpl_toolkits.mplot3d import Axes3D
                ax = plt.subplot(1, 2, 2, projection='3d')
                ax.plot(Zp_target[:, 0], Zp_target[:, 1], Zp_target[:, 2], '-o', markersize=3, alpha=0.6, label='Target')
                ax.plot(Zp_pred[:, 0], Zp_pred[:, 1], Zp_pred[:, 2], '-o', markersize=3, alpha=0.6, label='Prediction')
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                ax.set_zlabel('PC3')
                ax.set_title('Latent Trajectories (Top 3 PCA)')
                ax.legend()
                plt.tight_layout()
                plt.savefig(f"./latent_predictor_pca.png")
                plt.close()
                from PIL import Image
                img = Image.open("./latent_predictor_pca.png").convert("RGB")
                img_tensor = torchvision.transforms.ToTensor()(img)
                writer.add_image("PCA/Latent_Comparison", img_tensor, global_step=i+1)


            # Save the best model
            if val_loss < best_loss:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': i
                }
                new_checkpoint_path = f"{config.diffusion.save_out_path}/{config.diffusion.save_name}_best_epoch{i}_loss_{val_loss}.pt"
                torch.save(checkpoint, new_checkpoint_path)
                print(f"Saved new best model with validation loss {val_loss}")

                # If a previous best model exists, remove it
                if best_epoch is not None:
                    old_checkpoint_path = f"{config.diffusion.save_out_path}/{config.diffusion.save_name}_best_epoch{best_epoch}_loss_{best_loss}.pt"
                    if os.path.exists(old_checkpoint_path):
                        os.remove(old_checkpoint_path)
                best_epoch = i
                best_loss = val_loss

            # Save a checkpoint every n epochs
            if i % config.diffusion.save_interval == 0:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': i
                }
                torch.save(checkpoint, f"{config.diffusion.save_out_path}/{config.diffusion.save_name}_epoch{i}.pt")

            scheduler.step()
    writer.close()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
