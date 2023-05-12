import os
import torch
import pandas as pd
from skimage import io
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import argparse
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn as nn
import cv2

os.makedirs("model", exist_ok=True)

epochs = 32
csv_data_path = 'XD'

class LSTM_Dataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.descriptions_dataframe = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.descriptions_dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,self.descriptions_dataframe.iloc[idx, 0])
        image = io.imread(img_name,as_gray=True)
        if image.dtype == 'float64':
            image = cv2.normalize(src=image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        descriptions = self.descriptions_dataframe.iloc[idx, 1:]
        descriptions = np.array([descriptions])
        
        descriptions = descriptions.astype('float').reshape(-1, 1)
        if self.transform:
            image = self.transform(image)
        return image, descriptions

cuda = True if torch.cuda.is_available() else False

transformed_dataset = LSTM_Dataset(csv_file=csv_data_path,
root_dir='XD',
transform=transforms.Compose([
#transforms.ToPILImage(),
#transforms.Resize(opt.img_size),
#transforms.ToTensor(),
#transforms.Normalize([0.5], [0.5])
]))

dataloader = torch.utils.data.DataLoader(
    transformed_dataset,
    batch_size=10,
    shuffle=True,
)

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


for epoch in range(epochs):
    for i, (imgs, descriptions) in enumerate(dataloader):
        print("Aqui se itera XD")
# Para cuando quieras guardar un modelo

#torch.save(LSTM, "model/generator.pt")