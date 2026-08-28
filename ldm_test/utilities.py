import torch
import torchaudio
import math
import soundfile as sf

# Compute the latens
def compute_latents(w_model, dataloader, batch_size, device):
    dataset_latents = []
    scale_list = []
    for idx,input_wav in enumerate(dataloader):
        with torch.no_grad():
            # input_wav = input_wav.contiguous().cuda() #[B, 1, T]: eg. [2, 1, 203760]
            input_wav = input_wav.contiguous().to(device) #[B, 1, T]: eg. [2, 1, 203760]
            #save input wave
            sf.write("./input_wave.wav", input_wav[0,0,:].cpu().numpy(), samplerate=w_model.sample_rate)
            # check that input is 3 secs long
            if input_wav.shape[-1] != w_model.sample_rate * 5:
                print("Skipping sample with wrong length: ", input_wav.shape[-1])
                continue
            # ---------- Run Model ----------
            frames = w_model.encode(input_wav)
            # audio = w_model.decode(frames)

            # sf.write("./output_wave.wav", audio[0,0,:].cpu().numpy(), samplerate=w_model.sample_rate)
            # print(img)
            z_tmp,scale_tmp,mu_tmp,logvar_tmp = frames[0]
            z = z_tmp
            scale = scale_tmp
            for i in range(1,len(frames)):
                z_tmp,scale_tmp,mu_tmp,logvar_tmp = frames[i]
                z = torch.cat((z, z_tmp), dim = -1)
                if scale is not None:
                    scale = torch.cat((scale, scale_tmp), dim = -1)

            # print(z.shape) 
            # frames = reconstruct_encodec_frames(z, scale)
            # audio_recon = w_model.decode(frames)
            # sf.write("./reconstructed_wave.wav", audio_recon[0,0,:].cpu().numpy(), samplerate=w_model.sample_rate)
            scale_list.append(scale)
            dataset_latents.append(z)
            # dataset_labels.append(labels)
    dataset_latents = torch.cat(dataset_latents,0)
    if scale_list[0] is not None:
        scale_list = torch.cat(scale_list,0)
    else:
        scale_list = None
    # dataset_labels = torch.cat(dataset_labels,0)
    # labels not so important now, but will be in future
    # print("--- Exported dataset sizes:\t",dataset_latents.shape,dataset_labels.shape)
    print("--- Exported dataset sizes:\t", dataset_latents.shape)
    return dataset_latents[:,:,:256], scale_list
    # return dataset_latents
    # return dataset_latents,dataset_labels

# Export the latents
def export_latents(w_model, train_dataloader, val_dataloader, batch_size, device):
    train_latents, train_scales = compute_latents(w_model,train_dataloader, batch_size, device)
    test_latents, test_scales= compute_latents(w_model,val_dataloader, batch_size, device)
    return train_latents,test_latents, train_scales, test_scales

def reconstruct_encodec_frames(latents, scale, seg_len=75):

    use_scale = True
    if scale is None:
        use_scale = False

    chunks = math.ceil(latents.shape[-1] / seg_len)

    frames = []
    for i in range(chunks):
        tmp_frame = []
        tmp_frame.append(latents[:,:,i*seg_len:(i+1)*seg_len]) # Emb
        if use_scale == False:
            scale=None
            tmp_frame.append(scale) # Scale
        else:
            tmp_frame.append(scale[:,i])
        tmp_frame.append(latents[:,:,i*seg_len:(i+1)*seg_len]) # Mu
        tmp_frame.append(latents[:,:,i*seg_len:(i+1)*seg_len]) # Sig
        frames.append(tmp_frame)

    return frames
