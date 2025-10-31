import torch

# Compute the latens
def compute_latents(w_model, dataloader, batch_size, device):
    dataset_latents = []
    for idx,input_wav in enumerate(dataloader):
        with torch.no_grad():
            input_wav = input_wav.contiguous().cuda() #[B, 1, T]: eg. [2, 1, 203760]
            # ---------- Run Model ----------
            frames = w_model.encode(input_wav)

            z_tmp,scale,mu,logvar = frames[0]
            z = z_tmp
            for i in range(1,len(frames)):
                z_tmp,_,_,_ = frames[i]
                z = torch.cat((z, z_tmp), dim = -1)
           
            dataset_latents.append(z)
            # dataset_labels.append(labels)
    print(len(dataset_latents))
    dataset_latents = torch.cat(dataset_latents,0)
    print(dataset_latents.shape)
    # dataset_labels = torch.cat(dataset_labels,0)
    # labels not so important now, but will be in future
    # print("--- Exported dataset sizes:\t",dataset_latents.shape,dataset_labels.shape)
    print("--- Exported dataset sizes:\t", dataset_latents.shape)
    return dataset_latents
    # return dataset_latents,dataset_labels

# Export the latents
def export_latents(w_model, train_dataloader, val_dataloader, batch_size, device):
    train_latents = compute_latents(w_model,train_dataloader, batch_size, device)
    test_latents= compute_latents(w_model,val_dataloader, batch_size, device)
    return train_latents,test_latents
