
import torch
import torch.nn as nn
import torchaudio
import hydra
from torchaudio.transforms import Spectrogram,MelSpectrogram
import customAudioDataset as data
from latent_dataloaders import make_latent_dataloaders, create_dataset

# hydra breaks when this is included??
# from frechet_audio_distance import FrechetAudioDistance

from model import EncodecModel
from utils import export_latents
from customAudioDataset import collate_fn
from recurrent_model import RNN_v2
from datetime import datetime

DEVICE = 'cpu'
AUDIO_DIR = "/Users/adees/Code/neural_granular_synthesis/datasets/ESC-50_SeaWaves/audio/samples/5secs/small_train"
# LOAD_PATH = "/Users/adees/Code/encodec_tests/encodecModels/vae_encodecModel/bs16_cut48000_length0_epoch10000_lr0.0001_beta0.0001.pt"
# LOAD_PATH = "/Users/adees/Code/encodec_tests/encodecModels/vae_encodecModel/bs16_cut48000_length0_epoch10000_lr0.0001_seg0.25.pt"
# LOAD_PATH = "/Users/adees/Code/encodec_tests/encodecModels/vae_encodecModel/bs16_cut48000_length0_epoch10000_lr0.0001_beta0.01.pt"
LOAD_PATH = "/Users/adees/Code/encodec_tests/encodecModels/vae_encodecModel/fsd_50k/bs16_cut48000_length0_epoch900_lr0.0001.pt"

# frechet = FrechetAudioDistance(
#     model_name="vggish",
#     # Do I need to resample these?
#     sample_rate=16000,
#     use_pca=False, 
#     use_activation=False,
#     verbose=False
# )

def safe_log(x, eps=1e-7):
    return torch.log(x + eps)

class spectral_distances(nn.Module):
    def __init__(self,stft_scales=[2048, 1024, 512, 256, 128], mel_scales=[2048, 1024], spec_power=1, mel_dist=True, log_dist=0, sr=16000, device="cpu"):
        super(spectral_distances, self).__init__()
        self.stft_scales = stft_scales
        self.mel_scales = mel_scales
        self.mel_dist = mel_dist
        self.log_dist = log_dist
        T_spec = []
        for scale in stft_scales:
            T_spec.append(Spectrogram(n_fft=scale,hop_length=scale//4,window_fn=torch.hann_window,power=spec_power).to(device))
        self.T_spec = T_spec
        if mel_dist:
            # print("\n*** training with MelSpectrogram distance")
            T_mel = []
            for scale in mel_scales:
                T_mel.append(MelSpectrogram(n_fft=scale,hop_length=scale//4,window_fn=torch.hann_window,sample_rate=sr,f_min=50.,n_mels=scale//4,power=spec_power).to(device))
            self.T_mel = T_mel
    
    def forward(self,x_inp,x_tar):
        loss = 0
        n_scales = 0
        for i,scale in enumerate(self.stft_scales):
            S_inp,S_tar = self.T_spec[i](x_inp),self.T_spec[i](x_tar)
            stft_dist = (S_inp-S_tar).abs().mean()
            loss = loss+stft_dist
            n_scales += 1
            if self.log_dist>0:
                loss = loss+(safe_log(S_inp)-safe_log(S_tar)).abs().mean()*self.log_dist
                n_scales += self.log_dist
        if self.mel_dist:
            for i,scale in enumerate(self.mel_scales):
                M_inp,M_tar = self.T_mel[i](x_inp),self.T_mel[i](x_tar)
                mel_dist = (M_inp-M_tar).abs().mean()
                loss = loss+mel_dist
                n_scales += 1
                if self.log_dist>0:
                    loss = loss+(safe_log(M_inp)-safe_log(M_tar)).abs().mean()*self.log_dist
                    n_scales += self.log_dist
        return loss/n_scales

def convert_audio(wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
    assert wav.dim() >= 2, "Audio tensor must have at least 2 dimensions"
    assert wav.shape[-2] in [1, 2], "Audio must be mono or stereo."
    *shape, channels, length = wav.shape
    if target_channels == 1:
        wav = wav.mean(-2, keepdim=True)
    elif target_channels == 2:
        wav = wav.expand(*shape, target_channels, length)
    elif channels == 1:
        wav = wav.expand(target_channels, -1)
    else:
        raise RuntimeError(f"Impossible to convert from {channels} to {target_channels}")
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav

@hydra.main(config_path='config', config_name='config')
def main(config):
    
    trainset = data.CustomAudioDataset(config=config)
    testset = data.CustomAudioDataset(config=config,mode='test')

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=config.datasets.batch_size,
        collate_fn = collate_fn,
        shuffle=False,
        )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=config.datasets.batch_size,
        collate_fn = collate_fn,
        shuffle=False,
        )

    model = EncodecModel._get_model(
        config.model.target_bandwidths, 
        config.model.sample_rate, 
        config.model.channels,
        causal=config.model.causal, model_norm=config.model.norm, 
        audio_normalize=config.model.audio_normalize,
        segment=config.model.segment, name=config.model.name,
        ratios=config.model.ratios,
    )

    # model = EncodecModel.encodec_model_48khz()

    model_checkpoint = torch.load(LOAD_PATH, map_location='cpu')
    model.load_state_dict(model_checkpoint['model_state_dict'])

    model.to(DEVICE)
    model.eval()

    # from torchinfo import summary
    # summary(model, (1, 2, 12000))
    # summary(model, (1, 2, 320*4))

    train_latents,test_latents,= export_latents(model,trainloader,testloader,config.datasets.batch_size)

    LOOKBACK = 150
    X_train, y_train = create_dataset(train_latents, lookback=LOOKBACK)
    test_X_train, test_y_train = create_dataset(test_latents, lookback=LOOKBACK)

    X_train = X_train.reshape(-1, X_train.shape[2], X_train.shape[3])
    y_train = y_train.reshape(-1, y_train.shape[2], y_train.shape[3])
    test_X_train = test_X_train.reshape(-1, test_X_train.shape[2], test_X_train.shape[3])
    test_y_train = test_y_train.reshape(-1, test_y_train.shape[2], test_y_train.shape[3])

    LATENT_SIZE = 128
    HIDDEN_SIZE = 128
    NO_RNN_LAYERS = 1
    # l_model = RNN_v1(LATENT_SIZE, HIDDEN_SIZE, LATENT_SIZE, NO_RNN_LAYERS)
    l_model = RNN_v2(LATENT_SIZE, HIDDEN_SIZE, LATENT_SIZE, NO_RNN_LAYERS)
    LEARNING_RATE = 0.01
    optimizer = torch.optim.Adam(l_model.parameters(), lr=LEARNING_RATE)
    # lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.25, total_iters=3*EPOCHS/4)

    loss_fn = nn.MSELoss()
    # Note the batch size here 
    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), shuffle=True, batch_size=config.datasets.batch_size)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_X_train, test_y_train), shuffle=True, batch_size=config.datasets.batch_size)


    for epoch in range(config.common.max_epoch):
        l_model.train()
        running_rmse = 0.0;
        for X_batch, y_batch in train_loader:
            y_pred = l_model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            running_rmse += (loss.detach().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # # Decay the learning rate
        # lr_scheduler.step()
        # new_lr = optimizer.param_groups[0]["lr"]
        # Validation
        if epoch % config.common.test_interval != 0:
            continue
        l_model.eval()
        running_val_rmse = 0.0
        with torch.no_grad():
            for val_X_batch, val_y_batch in test_loader:
                y_pred = l_model(val_X_batch)
                running_val_rmse += (loss_fn(y_pred, val_y_batch))


        # recon_latent = val_X_batch[0,:,:]
        # # Add the next 10 samples on
        # tmp = val_X_batch[0,:,:].unsqueeze(0)
        # for i in range(0, y_pred.shape[0]):
        #     tmp = l_model(tmp)
        #     recon_latent = torch.cat((recon_latent, tmp[0,-1,:].unsqueeze(0)), dim=0)
        #     # recon_latent = torch.cat((recon_latent, y_pred[i,-1,:].unsqueeze(0)), dim=0)

        # print(recon_latent.shape)
        # print(test_latents.shape)
         
        # sampled_seq_loss = loss_fn(recon_latent[LOOKBACK:, :] ,test_latents[0, LOOKBACK:, :])

        train_rmse = running_rmse/len(train_loader)
        val_rmse = running_val_rmse/len(test_loader)

        print("Epoch %d: train RMSE %.16f, validation RMSE %.16f" % (epoch, train_rmse, val_rmse))
        # print(f"Seq Loss: {sampled_seq_loss}")

        #  # Early stopping Criteria
        #  if sampled_seq_loss < 0.0001 and EARLY_STOPPING:
        #      print("Stopped Early as test RMSE %.4f < 0.0001" % (sampled_seq_loss))
        #      torch.save({
        #              'epoch': epoch,
        #              'model_state_dict': l_model.state_dict(),
        #              'optimizer_state_dict': optimizer.state_dict(),
        #              'loss': train_rmse,
        #              }, f"{SAVE_MODEL_DIR}/latent_vae_latest_earlyStop.pt")
        #      break


        if (epoch) % config.common.save_interval == 0:
            torch.save({
                'epoch': epoch,
                # 'warmup_start': warmup_start,
                'model_state_dict': l_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_rmse,
                }, f"./latent_vae_{DEVICE}_{config.common.max_epoch}epochs_{config.datasets.batch_size}batch_{epoch}epoch_{datetime.now()}.pt")
            # Save as latest also
            torch.save({
                'epoch': epoch,
                'model_state_dict': l_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_rmse,
                }, f"./latent_vae_latest.pt")

     # Save after final epoch
    torch.save({
         'epoch': epoch,
         'model_state_dict': l_model.state_dict(),
         'optimizer_state_dict': optimizer.state_dict(),
         'loss': train_rmse,
         }, f"./latent_vae_latest.pt")

    print(train_latents.shape)
    print(img)

    with torch.no_grad():

        full_audio, sr_orig = torchaudio.load(AUDIO_DIR+"/1-39901-A-11.wav")
        # full_audio, sr_orig = torchaudio.load("/Users/adees/Code/pytorch-encodec-fork/data/noisebandComparison/real/audio.wav")
        input_wav = convert_audio(full_audio, sr_orig, model.sample_rate, model.channels)
        input_wav = input_wav.unsqueeze(0)

        # input_wav = input_wav[:,:,:2048]

        print(input_wav.shape)

        output, frames = model(input_wav)

        torchaudio.save(f'./recon.wav', output[0].cpu(), model.sample_rate, channels_first=True)
        torchaudio.save(f'./original.wav', input_wav[0].cpu(), model.sample_rate, channels_first=True)

        spec_dist = spectral_distances(sr=model.sample_rate, device=DEVICE)
        spec_loss = spec_dist(output, input_wav)
            # spec_loss = spec_dist(recon_audio, waveforms)
            # spec_loss_cat = spec_dist(recon_cat, waveforms)

        print("Spectral Loss: ", spec_loss)

        # fad_score = frechet.score(f'./original.wav', f'./recon.wav', dtype="float32")
        # print(f"FAD Score: {fad_score}")

if __name__ == '__main__':
    main()