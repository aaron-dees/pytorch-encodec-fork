import torch
import torchaudio
import math

# Compute the latens
def compute_latents(w_model, dataloader, batch_size, device):
    dataset_latents = []
    for idx,input_wav in enumerate(dataloader):
        with torch.no_grad():
            # input_wav = input_wav.contiguous().cuda() #[B, 1, T]: eg. [2, 1, 203760]
            input_wav = input_wav.contiguous().to(device) #[B, 1, T]: eg. [2, 1, 203760]
            # ---------- Run Model ----------
            frames = w_model.encode(input_wav)

            z_tmp,scale,mu,logvar = frames[0]
            z = z_tmp
            for i in range(1,len(frames)):
                z_tmp,_,_,_ = frames[i]
                z = torch.cat((z, z_tmp), dim = -1)
           
            dataset_latents.append(z)
            # dataset_labels.append(labels)
    dataset_latents = torch.cat(dataset_latents,0)
    # dataset_labels = torch.cat(dataset_labels,0)
    # labels not so important now, but will be in future
    # print("--- Exported dataset sizes:\t",dataset_latents.shape,dataset_labels.shape)
    print("--- Exported dataset sizes:\t", dataset_latents.shape)
    return dataset_latents[:,:,:256]
    # return dataset_latents
    # return dataset_latents,dataset_labels

# Export the latents
def export_latents(w_model, train_dataloader, val_dataloader, batch_size, device):
    train_latents = compute_latents(w_model,train_dataloader, batch_size, device)
    test_latents= compute_latents(w_model,val_dataloader, batch_size, device)
    return train_latents,test_latents

def reconstruct_encodec_frames(latents, seg_len=75):

    chunks = math.ceil(latents.shape[-1] / seg_len)

    frames = []
    for i in range(chunks):
        tmp_frame = []
        tmp_frame.append(latents[:,:,i*seg_len:(i+1)*seg_len]) # Emb
        scale = torch.Tensor(latents.shape[0],1).to(latents.device)
        scale[0,0]=0.1
        tmp_frame.append(scale) # Scale
        tmp_frame.append(latents[:,:,i*seg_len:(i+1)*seg_len]) # Mu
        tmp_frame.append(latents[:,:,i*seg_len:(i+1)*seg_len]) # Sig
        frames.append(tmp_frame)

    return frames
