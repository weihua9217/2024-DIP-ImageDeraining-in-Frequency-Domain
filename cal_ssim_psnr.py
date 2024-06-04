import os
from ldgp_freq import cal_ssim
from ldgp_freq import PSNR
import cv2

target_path = './dataset/Rain100L/target'
input_path = './dataset/Rain100L/input'
result_path = './dataset/Rain100L/avg_mul(fuse)'


before_psnr = 0
before_ssim = 0
after_psnr = 0
after_ssim =0
folder = os.listdir(input_path)
for file in folder:
   rain = cv2.imread(os.path.join(input_path, file))
   target = cv2.imread(os.path.join(target_path, file))
   result = cv2.imread(os.path.join(result_path, file))
   before_ssim += cal_ssim(rain, target)
   after_ssim += cal_ssim(result, target)

   before_psnr += PSNR(rain, target)
   after_psnr += PSNR(result, target)

print(before_psnr/len(folder))
print(after_psnr/len(folder))
print(before_ssim/len(folder))
print(after_ssim/len(folder))