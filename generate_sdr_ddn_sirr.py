import os 
import cv2
import numpy as np
import copy
import argparse
from ldgp_freq import PSNR
from ldgp_freq import cal_ssim
import time

parser = argparse.ArgumentParser()
parser.add_argument("--rainy_data_path", type=str, default="./dataset/DDN_SIRR/input/", help='Path to rainy data')
parser.add_argument("--ldgp_data_path", type=str, default="./dataset/DDN_SIRR/ldgp/", help='Path to ldgp data')
parser.add_argument("--sdr_result_path", type=str, default="./dataset/DDN_SIRR/sdr/", help='Path to save sdr data')
parser.add_argument("--gt_path", type=str, default="./dataset/DDN_SIRR/input/", help='Path to save sdr data')
parser.add_argument("--kernel_size", type=int, default=7, help='K')
parser.add_argument("--num", type=int, default=50, help='The numer of sdr for each images')
opt = parser.parse_args()

ks= opt.kernel_size
padding_num = ks//2
random_num = opt.num
input_path = opt.rainy_data_path
mask_path = opt.ldgp_data_path
result_path = opt.sdr_result_path
gt_path = opt.gt_path

try:
    os.mkdir(result_path)
except:
    pass

input_folder = os.listdir(input_path)
ldgp_folder = os.listdir(mask_path)

before_derain_psnr = 0
after_derain_psnr = 0
before_derain_ssim = 0
after_derain_ssim =0
total_time = 0

for idx in range(len(input_folder)):    
    print("Stochastic Derained References: ", input_folder[idx])
    start = time.process_time()
    # Create Folder to place sdr img
    dir_path = os.path.join(result_path,input_folder[idx][:-4])
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # Read Image
    img = cv2.imread(os.path.join(input_path, input_folder[idx]))
    mask = cv2.imread(os.path.join(mask_path, input_folder[idx]), cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(os.path.join(gt_path, input_folder[idx]))
    mask = cv2.bitwise_not(mask)
    h = mask.shape[0]
    w = mask.shape[1]
    original_psnr = PSNR(img, gt)
    original_ssim = cal_ssim(img, gt)
    
    # Create images w/ padding
    images = []
    for n in range(random_num):
        images.append( cv2.copyMakeBorder(copy.deepcopy(img),padding_num,padding_num,padding_num,padding_num, cv2.BORDER_CONSTANT, value=0))
    mask = cv2.copyMakeBorder(mask, padding_num,padding_num,padding_num,padding_num, cv2.BORDER_CONSTANT, value=0)
    img = cv2.copyMakeBorder(img, padding_num,padding_num,padding_num,padding_num, cv2.BORDER_CONSTANT, value=0)
    
    # Start to replace
    for j in range(padding_num, mask.shape[0]-padding_num+1):
        for i in range(padding_num, mask.shape[1]-padding_num+1):
            if mask[j,i] < 255:
                c_m = mask[j-padding_num:j+padding_num+1,i-padding_num:i+padding_num+1]
                neighbor = []
                for c_m_j in range(c_m.shape[0]):
                    for c_m_i in range(c_m.shape[1]):
                        if c_m[c_m_j,c_m_i]==255:
                            neighbor.append(ks*c_m_j+c_m_i)
                try:sample = np.random.choice(neighbor, random_num)
                except:break
                
                for n in range(random_num):
                    pix = sample[n]
                    images[n][j,i,:] = img[j+(pix//ks-padding_num),i+(pix%ks-padding_num),:]
                
    for l in range(random_num):
        cv2.imwrite(os.path.join(dir_path,input_folder[idx][:-4]+"-"+str(l)+'.png'),images[l][padding_num:padding_num+h, padding_num:padding_num+w])            
    
    all = np.array(images)
    fuse = np.average(all[:, padding_num:padding_num+h, padding_num:padding_num+w], axis=0, keepdims=False)
    # print(fuse.shape)
    # cv2.imwrite('./fuse.png', fuse)
    derain_psnr = PSNR(fuse.astype(np.uint8), gt.astype(np.uint8))
    derain_ssim = cal_ssim(fuse.astype(np.uint8), gt.astype(np.uint8))


    print(original_psnr, derain_psnr)
    print(original_ssim, derain_ssim)

    before_derain_psnr += original_psnr
    after_derain_psnr += derain_psnr
    before_derain_ssim += original_ssim
    after_derain_ssim += derain_ssim

    stop = time.process_time()  
    duration = stop - start
    total_time += duration


print(before_derain_psnr/len(input_folder))
print(after_derain_psnr/len(input_folder))

print(before_derain_ssim/len(input_folder))
print(after_derain_ssim/len(input_folder))

print("Average Time: ", total_time/len(input_folder))