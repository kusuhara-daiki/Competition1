import sys

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torchvision import transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.append( "./utils" )
import utils
from dataset.load_data import Load_Data
from dataset.image_process import set_img_process

class SEGDataset(Dataset, Load_Data):
    """
    データセットを作成するクラス
    """
    
    def __init__(self, cfg, args):
        super().__init__(cfg, args)

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        '''
        一個の画像と一個のラベルを返す関数
        index: int
        --------------------
        img:array
        label:int
        '''
        img, label = self.make_processed_train()
        img = img[index]
        label = label[index]
        return img,label
            
    def create_batches_with_labels(self, batch_size=None):
        """
        計算コスト軽減のためにバッチサイズごとに前処理を行う関数
        """
        if batch_size is None:
            batch_size = self.cfg.param.batch_size
        
        imgs_list = []
        labels_list = []
        
        num_samples = len(self.y_train)
        num_batches = num_samples // batch_size

        # バッチサイズで割り切れない場合は余りの部分を無視する
        images = self.x_train[:num_batches * batch_size]
        labels = self.y_train[:num_batches * batch_size]

        # バッチに変換
        batched_images = images.reshape((num_batches, batch_size, *images.shape[1:]))
        batched_labels = labels.reshape((num_batches, batch_size))
        
        for idx, (batch_images, batch_labels) in tqdm(enumerate(zip(batched_images, batched_labels)), desc='batch_process'):
            
            batch_processed_data = set_img_process(
                self.args, 
                batch_images, 
            )

            imgs_list.append(batch_processed_data)
            labels_list.append(batch_labels)
        
        processed_imgs = np.array(imgs_list)
        processed_labels = np.array(labels_list)

        dice3_processed_data = set_img_process(
            self.cfg,
            self.args, 
            self.dice3_imgs, 
        )
        
        imgs_array = np.concatenate([processed_imgs, dice3_processed_data], axis=0)
        labels_array = np.concatenate([processed_labels, self.dice3_labels], axis=0)
        
        return imgs_array, labels_array

                
    def make_processed_train(self):
        """
        画像の前処理を行い、保存する関数
        """
        
        processed_data = set_img_process(
            self.cfg,
            self.args, 
            self.x_train, 
        )
        
        dice3_processed_data = set_img_process(
            self.cfg,
            self.args, 
            self.dice3_imgs, 
        )
        
        
        imgs_array = np.concatenate([processed_data, dice3_processed_data], axis=0)
        labels_array = np.concatenate([self.y_train, self.dice3_labels], axis=0)
        
        if self.cfg.switch.flag.all_processed_imgs_labels_save == True:
            np.save('/work/data/processed_original/avg_all_processed_imgs', imgs_array)
            np.save('/work/data/processed_original/avg_all_processed_labels', labels_array)
        else:
            pass
            
        return imgs_array, labels_array

    def make_train_df(self, all_processed_imgs, all_processed_labels):
        """
        データローダーを作成する関数
        """
        
        explanatory_train, explanatory_valid, objective_train, objective_valid = train_test_split(all_processed_imgs, all_processed_labels, train_size=0.8, random_state=123)
    
        explanatory_train = torch.tensor(explanatory_train, dtype=torch.float32)
        explanatory_valid = torch.tensor(explanatory_valid, dtype=torch.float32)
        objective_train = torch.tensor(objective_train, dtype=torch.int64)
        objective_valid = torch.tensor(objective_valid, dtype=torch.int64)
        
        ds_train = TensorDataset(explanatory_train, objective_train)
        ds_valid = TensorDataset(explanatory_valid, objective_valid)
        
        train_loader = DataLoader(ds_train, batch_size=self.cfg.param.batch_size, shuffle=True)
        test_loader = DataLoader(ds_valid, batch_size=self.cfg.param.batch_size, shuffle=False)
    
        return train_loader, test_loader
    
    def transform(self):
        """
        tensorに対して画像の前処理を行う関数
        """
        
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
            ])
        
        test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)   
            ])
        
        train_resize = transforms.Compose([
            transforms.Resize((256,256))
        ])
        
        return train_resize



if __name__ == "__main__":
    
    args = utils.argparser()
    cfg = utils.config(args)
    
    segdataset = SEGDataset(cfg, args)
    
    all_processed_imgs, all_processed_labels = segdataset.make_processed_train()
        
    imgs, labels = segdataset.make_processed_train()
        
    
        