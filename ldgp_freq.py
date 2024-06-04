import cv2
import os
from ldgp import calculate_degree
from utils import set_line_to_values
import argparse
from matplotlib import pyplot as plt
import numpy as np
import time
from math import log10, sqrt 
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim

def cal_ssim(img1, img2) :
    # print(img1.shape, img2.shape)
    ssim_score, dif = ssim(img1, img2, full=True, channel_axis=-1, data_range=255)
    return ssim_score

def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 


def show(img, cvt=True):
   if cvt:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   plt.imshow(img)
   plt.show()

def cvt(img):
   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   return img

def show_gray(img):
   plt.imshow(img, cmap='gray')
   plt.show()


if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument("--dataset", type=str, default="Rain100L", help='Dataset')
   parser.add_argument("--patch_size", type=int, default=80 , help='Patch Size')
   parser.add_argument("--kernel_size", type=int, default=10, help='Kernel size')
   parser.add_argument("--thick", type=int, default=2, help='Kernel size')
   opt = parser.parse_args()

   input_path = os.path.join('dataset', opt.dataset, 'input')
   ldgp_path = os.path.join('dataset', opt.dataset, 'ldgp') # require to run generate_ldgp.py
   target_path = os.path.join('dataset', opt.dataset, 'target')

   result_path = os.path.join('./freq_result', opt.dataset)
   os.makedirs(result_path, exist_ok=True)

   folder = os.listdir(input_path)

   output_path = os.path.join('output', opt.dataset)
   os.makedirs(output_path, exist_ok=True)
   total_time = 0


   before_psnr = 0
   before_ssim = 0
   after_psnr = 0
   after_ssim = 0

   draw = True
   
   for img in folder:
      print(img)
      start = time.process_time()
      input_img = cv2.imread(os.path.join(input_path, img))
      target_img = cv2.imread(os.path.join(target_path, img))
      H, W = input_img.shape[:2]
      degree = calculate_degree(input_img)

      # 輸入圖轉到頻率域
      input_fshift_list = []
      input_freq_list = []
      for c in range(3):
         input_img_this = input_img[:,:,c]
         f = np.fft.fft2(input_img_this.astype(np.float32))
         input_fshift = np.fft.fftshift(f)
         input_fshift_list.append(input_fshift)
         # for 視覺化
         input_freq = np.log(np.abs(input_fshift))
         input_freq_list.append(input_freq)


      # 處理 ldgp 圖
      ldgp_img = cv2.imread(os.path.join(ldgp_path, img), 0)
      f = np.fft.fft2(ldgp_img.astype(np.float32))
      ldgp_fshift = np.fft.fftshift(f)
      # for 視覺化
      ldgp_freq = np.log(np.abs(ldgp_fshift))

      # 將輸入圖做處理
      fshift_results = []
      for c in range(3):
         fshift_result = input_fshift_list[c].copy()
         thick = opt.thick
         set_line_to_values(fshift_result, (W//2, H//2), degree, thick, values=0.75)
         c_t = 5
         fshift_result[H//2-c_t:H//2+c_t+1,:] = input_fshift_list[c][H//2-c_t:H//2+c_t+1,:]
         fshift_results.append(fshift_result)

      result_freq = np.log(np.abs(fshift_results[0]))
      
      # invert
      all_iimg = []
      for c in range(3):
         ishift = np.fft.ifftshift(fshift_results[c])
         iimg = np.fft.ifft2(ishift)
         iimg = np.abs(iimg)
         iimg[iimg>255]=255
         all_iimg.append(iimg.astype(np.uint8))

      color_imag = np.stack(all_iimg, axis=2)
      
      # calculate psnr
      
      stop = time.process_time()  
      duration = stop - start
      total_time += duration

      original_psnr = PSNR(input_img, target_img)
      original_ssim = cal_ssim(input_img, target_img)
      before_psnr+=original_psnr
      before_ssim+=original_ssim

      derain_psnr = PSNR(color_imag, target_img)
      derain_ssim = cal_ssim(color_imag, target_img)
      after_psnr+=derain_psnr
      after_ssim+=derain_ssim
      cv2.imwrite(os.path.join(result_path, img), color_imag)
      if draw:

         fig = plt.figure(figsize=(10, 4)) 
         fig.add_subplot(2, 3, 1)
         plt.imshow(cvt(input_img.astype(np.uint8))) 
         plt.axis('off') 
         plt.title("PSNR:%f"%original_psnr)

         fig.add_subplot(2, 3, 4)
         plt.imshow(input_freq_list[0], cmap='gray') 
         plt.axis('off') 
         
         fig.add_subplot(2, 3, 2)
         plt.imshow(cvt(color_imag.astype(np.uint8))) 
         plt.axis('off') 
         plt.title("PSNR:%f"%derain_psnr)
         
         
         fig.add_subplot(2, 3, 5)
         plt.imshow(result_freq, cmap='gray')
         plt.axis('off') 
         
         
         fig.add_subplot(2, 3, 3)
         plt.imshow(ldgp_img, cmap='gray')
         plt.axis('off') 
         
         fig.add_subplot(2, 3, 6)
         plt.imshow(ldgp_freq, cmap='gray')
         plt.axis('off') 

         # save image
         plt.savefig(os.path.join(output_path, img))
      

   print('Average time: ', total_time/len(folder))
   print('Before psnr: ', before_psnr/len(folder))
   print('After psnr: ', after_psnr/len(folder))

   print('Before ssim: ', before_ssim/len(folder))
   print('After ssim: ', after_ssim/len(folder))