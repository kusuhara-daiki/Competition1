import numpy as np
import pandas as pd
import torch

class Load_Data:
    """
    データを読み込むクラス
    """
    
    def __init__(self, cfg, args):
        self.args = args
        self.cfg = cfg
    
        self.x_train_origin = np.array(np.load(args.x_train_path), copy = True)

        self.x_train = self.x_train_origin.reshape(self.x_train_origin.shape[0], 1, 20, 20)
        
        self.x_test = np.load(args.x_test_path).copy()
        self.x_test = self.x_test.reshape(self.x_test.shape[0], 1, 20, 20)
        
        self.y_train = np.load(args.y_train_path).copy() - 1
        
        self.sample_submit = pd.read_csv(args.sample_submit_path).copy()
        
        self.dice3_imgs = np.load(args.dice3_imgs_path).copy()
        self.dice3_labels = np.load(args.dice3_labels_path).copy() - 1
        
        self.all_processed_imgs = np.load(args.all_processed_imgs_path).copy()
        self.all_processed_labels = np.load(args.all_processed_labels_path).copy()
        
        self.model = torch.load(args.after_model_path).to(cfg.mode.device.num)