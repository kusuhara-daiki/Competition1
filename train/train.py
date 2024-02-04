import torch

from tqdm import tqdm

from loss.loss import Focal_MultiLabel_Loss

def train_func(cfg, model, loader_train, optimizer, device):
    model.train()

    train_loss = 0
    n_train_acc = 0
    n_train = 0
    bar = tqdm(loader_train, desc="train")
    
    f_loss = Focal_MultiLabel_Loss(gamma=cfg.param.gamma_of_focal_loss)
    
    for images, labels in bar:
        train_batch_size = len(labels)
        n_train += train_batch_size
        
        optimizer.zero_grad()
        
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        
        loss = f_loss(outputs, labels)
        
        loss.backward()
        
        optimizer.step()
        
        predicted = torch.max(outputs, 1)[1]

        # 平均前の損失と正解数の計算
        # lossは平均計算が行われているので平均前の損失に戻して加算
        train_loss += loss.item() * train_batch_size  
        n_train_acc += (predicted == labels).sum().item()

    return train_loss, n_train_acc, n_train