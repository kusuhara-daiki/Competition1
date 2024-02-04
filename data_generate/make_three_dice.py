import numpy as np
import cv2
import random

from tqdm import tqdm
from skimage import transform
from skimage import filters

import utils
from dataset.load_data import Load_Data

class Make_Three_Dice(Load_Data):
    """
    サイコロ三つの画像を生成するクラス
    """
    
    def __init__(self, cfg, args):
        
        super().__init__(cfg, args)
        
        self.dict1, self.dict2 = self.make_one_two_dice()
        self.concat_dict = self.avg_concat_dice()
        self.reduction_dict = self.reduction()
        self.padded_dict = self.padding_dict()
        self.resize_20_dict = self.resize_20_three_dice()
        
        if self.args.make_three_dice:
            
            np.save('/work/data/processed_original/avg_dice3_imgs.npy', self.resize_20_dict['resize_20_imgs'])
            np.save('/work/data/processed_original/avg_dice3_labels.npy', self.resize_20_dict['resize_20_labels'])
            
        # self.noise_dict = self.make_noise()
        # self.resized_dict = self.make_resize_dict()
                
    def make_one_two_dice(self):
        """
        サイコロ一個とサイコロ二個の画像を白い部分の面積を基準にして分ける関数
        """
        
        dataset_dict = {
            'images': self.x_train_origin,
            'labels': self.y_train
            }
        
        imgs1 = []
        labels1 =[]
        imgs2 = []
        labels2 = []
        for i in tqdm(range(len(dataset_dict['labels'])), desc = 'separate_dice'):
            img = dataset_dict['images'][i]
            label = dataset_dict['labels'][i]
            processed_img = np.where(img > 30, 1, 0)
            
            if np.sum(processed_img, axis=0) < 85:
                imgs1.append(img)
                labels1.append(label)    
            else:
                imgs2.append(img)
                labels2.append(label)
                
        imgs1, labels1 = np.array(imgs1), np.array(labels1)
        imgs2, labels2 = np.array(imgs2), np.array(labels2)
        imgs1 = imgs1.reshape(imgs1.shape[0], 1, 20, 20)
        imgs2 = imgs2.reshape(imgs2.shape[0], 1, 20, 20)
        
        dict1 = {
            'images1': imgs1,
            'labels1': labels1
            }
        dict2 = {
            'images2': imgs2,
            'labels2': labels2
            }
        
        return dict1, dict2
    
    def avg_concat_dice(self):
        """
        サイコロ一個とサイコロ二個を繋げ、
        元画像のデータの不均衡をなくすために13以上の目のサイコロの数を均等に増やす関数
        """
        
        dice_under12_imgs_list =[]
        dice_under12_labels_list = []
        dice13_imgs_list = []
        dice13_labels_list = [] 
        dice14_imgs_list = []
        dice14_labels_list = []
        dice15_imgs_list = []
        dice15_labels_list = []
        dice16_imgs_list = []
        dice16_labels_list = []
        dice17_imgs_list = []
        dice17_labels_list = []
        dice18_imgs_list = []
        dice18_labels_list = []
        
        while len(dice13_imgs_list)<=3000 or len(dice14_imgs_list)<=3000 or len(dice15_imgs_list)<=3000 or len(dice16_imgs_list)<=3000:
            #  or len(dice17_imgs_list)<=3000 or len(dice18_imgs_list)<=3000
            num_samples = 10000
            random_num1 = random.sample(range(len(self.dict1['labels1'])), num_samples)
            random_num2 = random.sample(range(len(self.dict2['labels2'])), num_samples)
            
            for i, (n1, n2) in tqdm(enumerate(zip(random_num1, random_num2)), desc='avg_concat_dice'):
                img1 = self.dict1['images1'][n1]
                label1 = self.dict1['labels1'][n1]
                img2 = self.dict2['images2'][n2]
                label2 = self.dict2['labels2'][n2]
                
                concat_img = np.concatenate((img1, img2), axis=2)
                concat_label = label1 + label2
                
                if concat_label == 13:
                    if len(dice13_imgs_list) <= 3100: 
                        dice13_imgs_list.append(concat_img)
                        dice13_labels_list.append(concat_label)
                    else:
                        pass
                elif concat_label == 14:
                    if len(dice14_imgs_list) <= 3100:
                        dice14_imgs_list.append(concat_img)
                        dice14_labels_list.append(concat_label)
                    else:
                        pass
                elif concat_label == 15:
                    if len(dice15_imgs_list) <= 3100:
                        dice15_imgs_list.append(concat_img)
                        dice15_labels_list.append(concat_label)
                    else:
                        pass
                elif concat_label == 16:
                    if len(dice16_imgs_list) <= 3100:
                        dice16_imgs_list.append(concat_img)
                        dice16_labels_list.append(concat_label)
                    else:
                        pass
                else:
                    if len(dice_under12_imgs_list) < 20000:
                        dice_under12_imgs_list.append(concat_img)
                        dice_under12_labels_list.append(concat_label)
                    else:
                        pass
            
        concat_imgs_list =  dice_under12_imgs_list + dice13_imgs_list +  dice14_imgs_list +  dice15_imgs_list +  dice16_imgs_list
        concat_labels_list =  dice_under12_labels_list + dice13_labels_list +  dice14_labels_list +  dice15_labels_list +  dice16_labels_list
            
        concat_imgs_array = np.array(concat_imgs_list)
        concat_labels_array = np.array(concat_labels_list)
        
        concat_dict = {
            'concat_imgs':concat_imgs_array,
            'concat_labels':concat_labels_array
        }
        
        return concat_dict
    
    def concat_dice(self):
        """
        サイコロ一個とサイコロ二個を繋げる関数。データの不均衡は考慮していない。
        """
        
        num_samples = 10000
        random_num1 = random.sample(range(len(self.dict1['labels1'])), num_samples)
        random_num2 = random.sample(range(len(self.dict2['labels2'])), num_samples)

        concat_img_list = []
        concat_label_list = []
        
        for i, (n1, n2) in tqdm(enumerate(zip(random_num1, random_num2)), desc='concat_dice'):
            img1 = self.dict1['images1'][n1]
            label1 = self.dict1['labels1'][n1]
            
            img2 = self.dict2['images2'][n2]
            label2 = self.dict2['labels2'][n2]
            
            concat_img = np.concatenate((img1, img2), axis=2)
            concat_label = label1 + label2
            
            concat_img_list.append(concat_img)
            concat_label_list.append(concat_label)

        concat_img_array = np.array(concat_img_list)  
        concat_label_array = np.array(concat_label_list)
        
        concat_dict = {
            'concat_imgs':concat_img_array,
            'concat_labels':concat_label_array
        }
        
        return concat_dict
    
    def carve(self, num):
        """
        シームカービングを行う関数。
        """
        
        carved_list = []
        
        for image in tqdm(self.concat_dict['concat_imgs'], desc='seam_carving'):
            mag = filters.sobel(image.astype("float"))
            carved = transform.seam_carve(image, mag, (20, 20), mode='constant', cval=1, num=num)
            carved = (carved * 255).astype('uint8')
            
            carved_list.append(carved)
            
        carved_array = np.array(carved_list)  

        carved_dict = {
            'carved_imgs':carved_array,
            'carved_labels':self.concat_dict['concat_labels']
        }
        
        return carved_dict
    
    def reduction(self):
        """
        seam_carvingが使えないので代替案として考えた関数
        画像の行と列を一つずつ確認していき、1以下の画素値が含まれている箇所を削除することで黒い部分を減らす。
        """
        
        reduction_list = []
        
        for image in tqdm(self.concat_dict['concat_imgs'], desc='reduction'):
            result_matrix = np.copy(image)
            
            rows_to_remove = []

            for i in range(result_matrix.shape[1]):
                img_row = result_matrix[:, i, :]
                contains_one_or_more = np.any(img_row > 1)

                if not contains_one_or_more:
                    rows_to_remove.append(i)
                else:
                    pass    
                
            result_matrix = np.delete(result_matrix, rows_to_remove, axis=1)
            
            columns_to_remove = []
            
            for j in range(result_matrix.shape[2]):
                img_column = result_matrix[:, :, j]
                contains_one_or_more = np.any(img_column > 1)

                if not contains_one_or_more:
                    columns_to_remove.append(j)
                else:
                    pass
                
            result_matrix = np.delete(result_matrix, columns_to_remove, axis=2)
            reduction_list.append(result_matrix)
        
        reduction_dict = {
            'reduction_imgs':reduction_list,
            'reduction_labels':self.concat_dict['concat_labels']
        }    
            
        return reduction_dict
    
    def padding_dict(self):
        """
        画像のパディングを行う関数
        """
        
        padded_list = []
        
        for idx, img in tqdm(enumerate(self.reduction_dict['reduction_imgs']), desc='padding'):
            _, rows, cols = img.shape
            target_size = max(rows, cols)
            
            padded_image = np.ones((target_size, target_size), dtype=img.dtype)
            padded_image[:rows, :cols] = img
            padded_image = padded_image[np.newaxis, :, :]
            
            padded_list.append(padded_image)

        padded_dict = {
            'padded_imgs':padded_list,
            'padded_labels':self.concat_dict['concat_labels']
        }  

        return padded_dict
    
    def resize_20_three_dice(self):
        
        resize_20_list = []
        
        for idx, img in tqdm(enumerate(self.padded_dict['padded_imgs']), desc='resize_20'):
            img = img.squeeze()
            img = cv2.resize(img, (20, 20), interpolation=cv2.INTER_LINEAR)
        
            
            resize_20_list.append(img)
        
        resize_20_array = np.array(resize_20_list)

        resize_20_dict = {
            'resize_20_imgs': resize_20_array,
            'resize_20_labels':self.concat_dict['concat_labels']
        }  

        return resize_20_dict
    
        

    def make_noise(self, min_noise=0 , max_noise=32, probability=0.8):
        """
        画像にガウシアンノイズを付与する関数
        """
        
        noise_list = []
        
        for idx, img in tqdm(enumerate(self.padded_dict['padded_imgs']), desc='make_noise'):
            img_row = img.shape[1]
            img_column = img.shape[2]
            img = img.reshape(1, -1)
            
            if np.random.rand() < probability:
                noise = np.random.normal(min_noise, max_noise, img.shape) 
                noisy_img = img.astype(np.float64) + noise
                noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
            else:
                noisy_img = img  
                
            noisy_img = noisy_img.reshape(1, img_row, img_column)
            
            noise_list.append(noisy_img)
            
        noise_dict = {
            'noise_imgs':noise_list,
            'noise_labels':self.concat_dict['concat_labels']
        }    
        
        return noise_dict
        
    
    def make_resize_dict(self):
        """
        resizeを行う関数
        """
        
        resize_list = []
        
        for idx, img in tqdm(enumerate(self.noise_dict['noise_imgs'])):
            resize_img = cv2.resize(img.transpose(1, 2, 0), (256, 256), interpolation=cv2.INTER_LINEAR)
            resize_img = np.expand_dims(resize_img, axis=0)
            resize_list.append(resize_img)
        resize_array = np.array(resize_list)
        
        resize_dict = {
            'resize_imgs':resize_array,
            'resize_labels':self.concat_dict['concat_labels']
        }
        
        return resize_dict

if __name__=='__main__':
    
    args = utils.argparser()
    cfg = utils.config(args)
    
    Make_Three_Dice(cfg, args)