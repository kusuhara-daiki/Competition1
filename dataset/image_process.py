import cv2
import csv

import numpy as np
import pandas as pd
import matplotlib as plt

from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

import utils
from dataset.load_data import Load_Data
class Image_processing(Load_Data):
    """
    さまざまな画像の前処理を実装するクラス
    """
    
    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        
    def resize_img(self, img):
        """
        resizeする関数
        """
        
        img = cv2.resize(img, (self.cfg.param.resize_size, self.cfg.param.resize_size), interpolation=cv2.INTER_LINEAR)
        
        return img

    def to_binary(self, img):
        """
        二値化する関数
        """
        
        img = img.copy()
        
        threshold = 128
        
        ret, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        
        return img

    def to_otsu_binary(self, img):
        """
        大津の二値化する関数
        """
        
        img = img.copy()
        
        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        
        return img
        
    def opening(self, img):
        """
        オープニング処理
        """
        
        kernel = np.ones((5,5),np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) 
                
        return img  
    
    def closing(self, img):
        """
        クロージング処理
        """
        
        kernel = np.ones((5,5),np.uint8)
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        
        return closing
    
    def connected_components(self, img):
        """
        連結成分をカウントする関数
        """
        
        n_label, labels = cv2.connectedComponents(img)
        
        return n_label, labels
    
    def find_contours(self,img):
        """
        サイコロの輪郭を抽出する関数
        """
        
        #cv2.RETR_TREEは輪郭の構造を階層で表示
        #cv2.CHAIN_APPROX_SIMPLEは輪郭を構成する座標をなるべく直線になるように表示
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # サイコロ同士の階層構造における親子関係を特定
        connected_dice = []
        for i, h in enumerate(hierarchy[0]):
            if h[3] == -1:
                # 親がない場合、つまりサイコロの輪郭の場合
                connected_dice.append(i)

        dice_list = []
        # サイコロ同士を分離
        for idx in connected_dice:
            mask = np.zeros_like(img)
            cv2.drawContours(mask, contours, idx, 255, thickness=cv2.FILLED)
            dice = cv2.bitwise_and(img, mask)
            
            dice_list.append(dice)
            
        return dice_list    
    
    def dice_rotation(self, img):
        """
        サイコロを回転させてそれぞれのサイコロの向きを揃える関数
        """
        
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # 輪郭を囲む矩形を取得
            # x, y:左上の座標 width height
            x, y, w, h = cv2.boundingRect(contour)

            # 矩形の中心を求める
            center_x, center_y = x + w // 2, y + h // 2

            # 矩形の角度を求める
            # 中心座標(x,y), (width, height), 回転角度
            _, _, angle = cv2.minAreaRect(contour)

            # 矩形の向きを揃える変換行列を作成
            rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), -angle, 1)

            # 画像を変換
            rotated_img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))
            
            return rotated_img
        
    def make_noise(self, img, min_noise=0 , max_noise=32, probability=0.8):
        """
        ノイズを付与する関数
        """
        
        img = img.transpose(2, 0, 1)
        img_row = img.shape[1]
        img_column = img.shape[2]
        img = img.reshape(1, -1)
        
        if np.random.rand() < probability:
            noise = np.random.normal(min_noise, max_noise, img.shape) 
            noisy_img = img.astype(np.float64) + noise
            noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        else:
            noisy_img = img  
            
        noisy_img = noisy_img.reshape(img_row, img_column, 1)  
        
        return noisy_img
    
    def reshape_img(self, img):
        """
        reshapeする関数
        """
        
        reshape_img = img.reshape(1, 20, 20)
        
        return reshape_img


def set_img_process(cfg,
                    args, 
                    imgs,
                    noise=None,
                    resize=None,
                    opening=None, 
                    closing=None,
                    binary=None
                    ):
    """
    行いたい画像処理をまとめて行うための関数
    """
    if noise is not None:
        noise = cfg.switch.process.noise
    if resize is not None:
        resize = cfg.switch.process.resize
    if opening is not None:
        opening = cfg.switch.process.opening
    if closing is not None:
        closing = cfg.switch.process.closing
    if binary is not None:
        binary = cfg.switch.process.binary
        
    image_processing = Image_processing(cfg, args)
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    
    processed_list = []
    
    for img in tqdm(imgs):
        if noise == True:
            img = image_processing.make_noise(img)
        if resize == True:   
            img = image_processing.resize_img(img)
        if opening == True:   
            img = image_processing.opening(img)
        if closing == True:      
            img = image_processing.closing(img)
        if binary == True:
            img = image_processing.to_otsu_binary(img)
        processed_list.append(img)
        
    processed_imgs = np.array(processed_list)[:, np.newaxis, :, :]
        
    return processed_imgs