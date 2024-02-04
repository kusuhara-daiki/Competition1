import os
import utils

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import datetime
import yaml

from tqdm import tqdm
from torch import optim

from dataset.dataset import SEGDataset
from model.model import choose_model
from train.train import train_func
from valid.valid import valid_func


"""
実行ファイル
"""

def main(cfg, args):
    device = torch.device(f"cuda:{cfg.mode.device.num}" if torch.cuda.is_available() else "cpu")  
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    # dataset クラスの定義
    segdataset = SEGDataset(cfg, args)
    # dataloader の定義
    train_dataloader = DataLoader(detaset)
    # valid_data_loader = Da//
    
    #モデル定義
    model = choose_model(cfg, args, device)
    
    #学習の設定
    lr = cfg.param.init_lr
    num_epochs = cfg.param.n_epochs
    momentum = cfg.param.momentum
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    history = np.zeros((0, 5))
    
    for epoch in tqdm(range(1, num_epochs+1), desc="epoch"):
        
        #学習
        train_loss, n_train_acc, n_train = train_func(cfg, model, train_loader, optimizer, device)
        val_loss, n_val_acc, n_test = valid_func(cfg, model, test_loader, device)
        
        # 精度計算
        train_acc = n_train_acc / n_train
        val_acc = n_val_acc / n_test
        
        # 損失計算
        avg_train_loss = train_loss / n_train
        avg_val_loss = val_loss / n_test
        
        # 結果表示
        print (f'Epoch [{(epoch)}/{num_epochs}], loss: {avg_train_loss:.5f} acc: {train_acc:.5f} val_loss: {avg_val_loss:.5f}, val_acc: {val_acc:.5f}')
        
        # 記録
        item = np.array([epoch, avg_train_loss, train_acc, avg_val_loss, val_acc])
        history = np.vstack((history, item))
        
        #モデルの保存
        result_file_name = f'model6_{epoch}.pth'
        torch.save(model, os.path.join(args.result_folder, result_file_name))
        

    # グラフのプロット
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Lossのヒストグラム
    ax1.plot(history[:, 0], history[:, 1], color='blue', alpha=0.7, label='avg_train_loss')
    ax1.plot(history[:, 0], history[:, 3], color='red', alpha=0.7, label='avg_val_loss')
    ax1.set_title('plot of Loss')
    ax1.set_xlabel('Loss')
    ax1.set_ylabel('Frequency')
    ax1.legend()

    # Accuracyのヒストグラム
    ax2.plot(history[:, 0], history[:, 2], color='blue', alpha=0.7, label='train_acc')
    ax2.plot(history[:, 0], history[:, 2], color='red', alpha=0.7, label='val_acc')
    ax2.set_title('plot of Accuracy')
    ax2.set_xlabel('Accuracy')
    ax2.set_ylabel('Frequency')
    ax2.legend()

    plt.tight_layout()

    plt.savefig(os.path.join(args.result_datetime_folder, f'history_plot{current_time}.png'))
    
    plt.close()
    
    np.savetxt(os.path.join(args.result_datetime_folder, f'history{current_time}.csv'), history, delimiter=',', header='epoch, avg_train_loss, train_acc, avg_val_loss, val_acc', comments='')
        

if __name__ == "__main__":
    
    args = utils.argparser()
    cfg = utils.config(args)
    
    main(cfg, args)
    