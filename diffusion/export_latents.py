import sys
sys.path.append("..")

import torch
import torch.nn as nn
import torchaudio
import hydra

from model import EncodecModel
import customAudioDataset as data
from customAudioDataset import collate_fn
from utilities import export_latents

LOAD_PATH = "/Users/adees/Code/encodec_tests/encodecModels/vae_encodecModel/es_seaWaves/1secSamples_bs16_cut48000_length0_epoch140_lr0.0003.pt"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@hydra.main(config_path='../config', config_name='config')
def main(config):

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

    # model_checkpoint = torch.load(LOAD_PATH, map_location='cpu')
    # model.load_state_dict(model_checkpoint['model_state_dict'])

    w_model.to(DEVICE)
    w_model.eval()

    print("model loaded")

    print("Exporting Latents...")

    train_latents, test_latents = export_latents(w_model, wav_trainloader, wav_testloader, config.datasets.batch_size, DEVICE)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_latents),
        shuffle=True,
        batch_size=config.datasets.batch_size,
        # num_workers=0,
        # pin_memory=False,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_latents),
        shuffle=True,
        batch_size=config.datasets.batch_size,
        # num_workers=0,
        # pin_memory=False,
    )

if __name__ == '__main__':
    main()
