import cv2

import matplotlib.pyplot as plt
import numpy as np

from dataset.load_data import Load_Data

class EDA(Load_Data):
    """
    データの解析を行うクラス
    """
    
    def __init__(self, cfg, args):
        super.__init__(cfg, args)
        
    def pixel_frequency_hist(self, img):
        """
        画素値の出現頻度をヒストグラムにする関数
        """
        
        img = cv2.imread(f'{img}').astype(np.float)
        
        plt.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))
        
        if self.cfg.pixel_frequency_hist_save == True:
            plt.savefig(f'{img}.png', formal='png')
        else:
            pass
        
        return img
    
    def surface_area_of_dice(self):
        """
        サイコロ一個とサイコロ二個の画像の白色の面積を比較する関数
        """
        
        x_train = np.load(self.x_train)
        binary_img = np.where(x_train > 30, 1, 0)
        white_area = np.sum(binary_img, axis=1).tolist()
        
        plt.hist(white_area, bins=20, color='blue', alpha=0.7, label='freq_menseki')
        
        if self.cfg.surface_area_of_dice_save == True:
            plt.savefig('/work/data/result/frq_hist.png')
        else:
            pass
    


