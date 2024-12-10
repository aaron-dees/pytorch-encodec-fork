
import torch
import torch.nn as nn
import torchaudio
import hydra
from torchaudio.transforms import Spectrogram,MelSpectrogram

# hydra breaks when this is included??
# from frechet_audio_distance import FrechetAudioDistance

from model import EncodecModel
from utils import _linear_overlap_add

DEVICE = 'cpu'
AUDIO_DIR = "/Users/adees/Code/neural_granular_synthesis/datasets/ESC-50_SeaWaves/audio/samples/5secs/small"
# LOAD_PATH = "/Users/adees/Code/encodec_tests/encodecModels/vae_encodecModel/bs16_cut48000_length0_epoch10000_lr0.0001_beta0.0001.pt"
LOAD_PATH = "/Users/adees/Code/encodec_tests/encodecModels/vae_encodecModel/bs16_cut48000_length0_epoch10000_lr0.0001_seg0.0625.pt"

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

    model = EncodecModel._get_model(
        config.model.target_bandwidths, 
        config.model.sample_rate, 
        config.model.channels,
        causal=config.model.causal, model_norm=config.model.norm, 
        audio_normalize=config.model.audio_normalize,
        segment=config.model.segment, name=config.model.name,
        ratios=config.model.ratios,
    )

    model_checkpoint = torch.load(LOAD_PATH, map_location='cpu')
    model.load_state_dict(model_checkpoint['model_state_dict'])

    # model = EncodecModel.encodec_model_48khz()


    model.to(DEVICE)
    model.eval()

    with torch.no_grad():

        full_audio, sr_orig = torchaudio.load(AUDIO_DIR+"/1-39901-A-11.wav")
        # full_audio, sr_orig = torchaudio.load("/Users/adees/Code/pytorch-encodec-fork/data/noisebandComparison/real/audio.wav")
        input_wav = convert_audio(full_audio, sr_orig, model.sample_rate, model.channels)
        input_wav = input_wav.unsqueeze(0)

        # input_wav = input_wav[:,:,:2048]

        # Get frames and latents etc...
        output, frames = model(input_wav)

        # Decode one frame at a time and overlap add them.
        decoded_frames = [model._decode_frame(frame) for frame in frames]
        print("Decoded frames: ", len(decoded_frames))
        output_single_frame = _linear_overlap_add(decoded_frames, model.segment_stride or 1)

        stride = 10

        # get the number of latent points in a frame.
        output_list = []
        for j in range(0, len(frames)):
            output_tmp = []
            n_latent_points = frames[j][0].shape[-1]
            for i in range(0, n_latent_points, stride):
                single_latent = frames[j][0][:,:,i:i+stride]
                single_scale = frames[j][1]
                single_mu = frames[j][2][:,:,i:i+stride]
                single_beta = frames[j][3][:,:,i:i+stride]

                # Noise testing
                single_latent += (torch.randn_like(single_latent)*2. - 1.)

                # single_tuple = (single_latent, single_scale)
                single_tuple = (single_latent, single_scale, single_mu, single_beta)

                new_frames = [single_tuple]

                tmp = model.decode(new_frames)

                output_tmp.append(tmp)

            output = torch.cat(output_tmp, dim=-1)
            output_list.append(output)

        output = _linear_overlap_add(output_list, model.segment_stride or 1)

        torchaudio.save(f'./recon_seq.wav', output[0].cpu(), model.sample_rate, channels_first=True)
        torchaudio.save(f'./recon_frame.wav', output_single_frame[0].cpu(), model.sample_rate, channels_first=True)
        torchaudio.save(f'./original.wav', input_wav[0].cpu(), model.sample_rate, channels_first=True)

        print(img)

        spec_dist = spectral_distances(sr=model.sample_rate, device=DEVICE)
        spec_loss = spec_dist(output, input_wav)
            # spec_loss = spec_dist(recon_audio, waveforms)
            # spec_loss_cat = spec_dist(recon_cat, waveforms)

        print("Spectral Loss: ", spec_loss)

        # fad_score = frechet.score(f'./original.wav', f'./recon.wav', dtype="float32")
        # print(f"FAD Score: {fad_score}")

if __name__ == '__main__':
    main()