import torch

def make_latent_dataloaders(train_latents,test_latents,batch_size,num_workers=2):
    train_dataset = torch.utils.data.TensorDataset(train_latents)
    test_dataset = torch.utils.data.TensorDataset(test_latents)
    # shapes are latents = [N,n_grains,z_dim] ; labels = [N]
    print("--- Latent train/test sizes: ",len(train_dataset),len(test_dataset))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)
    return train_dataloader,test_dataloader

def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    feature = dataset[:, 0:0+lookback,:].unsqueeze(1)
    target = dataset [:, 0+1:0+lookback+1,:].unsqueeze(1)
    for i in range(1, dataset.shape[1]-lookback):
        feature = torch.cat((feature, dataset[:,i:i+lookback,:].unsqueeze(1)),1)
        target = torch.cat((target, dataset[:,i+1:i+lookback+1,:].unsqueeze(1)),1)

    return feature, target