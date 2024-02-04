
import torch
import tensorflow as tf
from torch import nn
import torch.nn.functional as F
from torch import nn

def loss_func(input, label):
    """
    クロスエントロピー関数を定義した関数
    """
    
    loss = nn.CrossEntropyLoss()
    loss = loss(input, label)
            
    return loss


def focal_loss(logits, targets, num_classes=18, gamma=2.0, alpha=None):
    """
    focal lossを計算する関数

    Args:
    - logits: ネットワークの出力 logits
    - targets: 正解ラベル
    - num_classes: クラス数
    - gamma: フォーカスパラメータ
    - alpha: クラスごとの重み（Noneの場合は均等に扱う）

    Returns:
    - loss: フォーカス損失
    """
    
    probs = logits
    
    ce_loss = nn.CrossEntropyLoss()
    ce_loss = ce_loss(logits, targets)
    
    # クラスごとの重みを設定
    if alpha is None:
        alpha = torch.ones(num_classes, device=logits.device)

    # ターゲットのone-hotエンコーディング
    one_hot_targets = F.one_hot(targets, num_classes=num_classes).float().to(logits.device)

    # ターゲットがポジティブ（正例）の場合は alpha を適用
    alpha_factor = alpha * one_hot_targets

    # ターゲットがネガティブ（負例）の場合は 1 - alpha を適用
    1 - alpha_factor

    # フォーカス損失の計算
    focal_weights = (1 - probs).pow(gamma)
    
    focal_loss = torch.sum(alpha_factor[:, None, :] * focal_weights[:, None, :] * ce_loss[:, None, :], dim=2)

    return focal_loss.mean()

class Focal_MultiLabel_Loss(nn.Module):
    """
    focal lossの実装
    上の関数が動かなかったのでこっちを使う
    """
    
    def __init__(self, gamma):
        super(Focal_MultiLabel_Loss, self).__init__()
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets): 
        ce = self.ce_loss(outputs, targets)
        ce_exp = torch.exp(-ce)
        focal_loss = (1-ce_exp)**self.gamma * ce
        return focal_loss.mean()

    
    
    
    