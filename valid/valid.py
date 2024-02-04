import torch

from tqdm import tqdm

from loss.loss import Focal_MultiLabel_Loss

def valid_func(cfg, model, valid_loader, device):
    model.eval()

    val_loss = 0
    n_val_acc = 0
    n_test = 0
    bar = tqdm(valid_loader, desc="valid")
    
    f_loss = Focal_MultiLabel_Loss(gamma=cfg.param.gamma_of_focal_loss)
    
    with torch.no_grad():
        for images, labels in bar:
            test_batch_size = len(labels)
            n_test += test_batch_size
            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            loss = f_loss(outputs, labels)
        
            predicted_num = torch.max(outputs, 1)[1]

            # 平均前の損失と正解数の計算
            # lossは平均計算が行われているので平均前の損失に戻して加算
            val_loss +=  loss.item() * test_batch_size
            n_val_acc +=  (predicted_num == labels).sum().item()
            
    return val_loss, n_val_acc, n_test