import numpy as np
import matplotlib.pyplot as plt
import cv2

from PIL import Image
from skimage import io
"""
シームカービングを実装したファイル
実行はできるが結果が悪い
"""

def energy_map(image):
    # Sobelフィルタを使用してエネルギーマップを計算
    # grayscale = color.rgb2gray(image)
    grayscale = image_gray
    sobel_x = np.abs(np.gradient(grayscale, axis=1))
    sobel_y = np.abs(np.gradient(grayscale, axis=0))
    energy = sobel_x + sobel_y
    return energy

def find_vertical_seam(energy):
    # 垂直方向のシームを見つける
    rows, cols = energy.shape
    seam = np.zeros(rows, dtype=int)
    seam[-1] = np.argmin(energy[-1, :])

    for i in range(rows - 2, -1, -1):
        j = seam[i + 1]
        choices = [j - 1, j, j + 1]
        choices = [c for c in choices if 0 <= c < cols]
        seam[i] = choices[np.argmin(energy[i, choices])]

    return seam

def remove_vertical_seam(image, seam):
    # 垂直方向のシームを取り除く
    rows, cols, _ = image.shape
    new_image = np.zeros((rows, cols - 1, 3), dtype=np.uint8)

    for i in range(rows):
        j = seam[i]
        new_image[i, :, :] = np.delete(image[i, :, :], j, axis=0)

    return new_image

def find_horizontal_seam(energy):
    rows, cols = energy.shape
    seam = np.zeros(cols, dtype=int)
    seam[-1] = np.argmin(energy[:, -1])

    for j in range(cols - 2, -1, -1):
        i = seam[j + 1]
        choices = [i - 1, i, i + 1]
        choices = [c for c in choices if 0 <= c < rows]
        seam[j] = choices[np.argmin(energy[choices, j])]

    return seam

def remove_horizontal_seam(image, seam):
    rows, cols, _ = image.shape
    new_image = np.zeros((rows - 1, cols, 3), dtype=np.uint8)

    for j in range(cols):
        i = seam[j]
        new_image[:, j, :] = np.delete(image[:, j, :], i, axis=0)

    return new_image

def seam_carve(image, num_seams):
    # シーム彫刻を実行
    resized_image = np.copy(image)

    for _ in range(num_seams):
        energy = energy_map(resized_image)
        seam = find_horizontal_seam(energy)
        resized_image = remove_horizontal_seam(resized_image, seam)

    return resized_image
            


if __name__ == "__main__":

    # 画像を読み込み
    img_path = '/work/data/image_folder/after_process/process_imgs/concat_imgs/concat_img0.png'
    image_gray = io.imread(img_path)
    image = np.expand_dims(image_gray, axis=2)
    
    image_copy = image.copy() 
    ret, image_otsu = cv2.threshold(image_copy, 0, 255, cv2.THRESH_OTSU)
    image_otsu = Image.fromarray(image_otsu)
    image_otsu.save(f'/work/data/image_folder/after_process/process_imgs/seam_carved_imgs/otsuimg.png')
    breakpoint()
    
    # シームカービングを実行
    num_seams_to_remove = 10
    resized_image = seam_carve(image, num_seams_to_remove)

    # 結果の表示
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_otsu)
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(resized_image)
    plt.title('Resized Image after Seam Carving')

    plt.savefig('/work/data/image_folder/after_process/process_imgs/seam_carved_imgs/otsu.png')
