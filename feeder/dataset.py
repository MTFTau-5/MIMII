import torch
from torch.utils.data import Dataset
import pickle
import torch.nn as nn

class AudioDataset(Dataset):
    def __init__(self, pkl_file_path, num_devices):
        with open(pkl_file_path, 'rb') as otto:
            all_data = pickle.load(otto)
        self.merged_data = []
        for snr_data in all_data:
            self.merged_data.extend(snr_data)
        self.num_devices = num_devices
        
        self.embedding_model = EmbeddingModel(num_devices)

    def __len__(self):
        return len(self.merged_data)



    def __getitem__(self, index):
        mfcc_feature = self.merged_data[index][0]
        device_num = self.merged_data[index][1]
        label = self.merged_data[index][2]
        mfcc_feature_tensor = torch.from_numpy(mfcc_feature).float()
        device_num_tensor = torch.tensor(device_num).long()
        
        device_num_embedding = self.embedding_model(device_num_tensor)
        
        return mfcc_feature_tensor, device_num_embedding, label





class EmbeddingModel(nn.Module):
    def __init__(self, num_devices):
        super(EmbeddingModel, self).__init__()
        self.device_num = nn.Embedding(num_devices, 11)

    def forward(self, device_num_tensor):
        
        device_num_embedding = self.device_num(device_num_tensor)
        return device_num_embedding