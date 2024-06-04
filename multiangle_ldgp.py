import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="DDN_SIRR", help='Dataset')
parser.add_argument("--patch_size", type=int, default=80 , help='Patch Size')
parser.add_argument("--kernel_size", type=int, default=10, help='Kernel size')
opt = parser.parse_args()

def hog(img, show= True, save="", cell_size = (8, 8), num_cells_per_block = (4, 4)): 
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    block_size = (num_cells_per_block[0] * cell_size[0],
              num_cells_per_block[1] * cell_size[1])
    x_cells = gray_img.shape[1] // cell_size[0]
    y_cells = gray_img.shape[0] // cell_size[1]
    h_stride = 1
    v_stride = 1
    block_stride = (cell_size[0] * h_stride, cell_size[1] * v_stride)
    num_bins = 36
    win_size = (x_cells * cell_size[0] , y_cells * cell_size[1])
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
    hog_descriptor = hog.compute(gray_img)
    
    tot_bx = np.uint32(((x_cells - num_cells_per_block[0]) / h_stride) + 1)
    tot_by = np.uint32(((y_cells - num_cells_per_block[1]) / v_stride) + 1)

    tot_els = (tot_bx) * (tot_by) * num_cells_per_block[0] * num_cells_per_block[1] * num_bins
    
    plt.rcParams['figure.figsize'] = [9.8, 9]

    hog_descriptor_reshaped = hog_descriptor.reshape(tot_bx,
                                                    tot_by,
                                                    num_cells_per_block[0],
                                                    num_cells_per_block[1],
                                                    num_bins).transpose((1, 0, 2, 3, 4))

    ave_grad = np.zeros((y_cells, x_cells, num_bins))
    hist_counter = np.zeros((y_cells, x_cells, 1))

    for i in range (num_cells_per_block[0]):
        for j in range(num_cells_per_block[1]):
            ave_grad[i:tot_by + i, j:tot_bx + j] += hog_descriptor_reshaped[:, :, i, j, :]
            
            hist_counter[i:tot_by + i, j:tot_bx + j] += 1

    ave_grad /= hist_counter
    sumofdegree = [0 for _ in range(num_bins)]
    
    for i in range(num_bins):
        for j in range(ave_grad.shape[0]):
            for k in range(ave_grad.shape[1]):
                sumofdegree[i]+=ave_grad[j][k][i]
    
    
    x = [(180/num_bins)*(i) for i in range(num_bins)]
    
    plt.bar(x,
        sumofdegree, 
        width=180/num_bins, 
        bottom=None, 
        align='center', 
        color=['lightsteelblue', 
            'cornflowerblue', 
            'royalblue', 
            'midnightblue', 
            'navy', 
            'darkblue', 
            'mediumblue'])
    plt.xticks(rotation='vertical')
    if save:
        plt.savefig(save)
    if show:
        plt.show()
    
    plt.cla()
    return sumofdegree.index(max(sumofdegree))

def most_frequent(List):
    return max(set(List), key = List.count)#把出現次數最多的取出
    
def rotate(img, theta):
    rows, cols = img.shape[0], img.shape[1]
    image_center = (cols/2, rows/2)

    M = cv2.getRotationMatrix2D(image_center,theta,1)

    abs_cos = abs(M[0,0])
    abs_sin = abs(M[0,1])

    bound_w = int(rows * abs_sin + cols * abs_cos)
    bound_h = int(rows * abs_cos + cols * abs_sin)

    M[0, 2] += bound_w/2 - image_center[0]
    M[1, 2] += bound_h/2 - image_center[1]
    rotated = cv2.warpAffine(img,M,(bound_w,bound_h),borderValue=(255,255,255))
    #把img 依照M的方式去旋轉並放到bound_w x bound_h的圖片，boderValue則是沒有值得地方要是什麼值，
    #應該是旋轉後會有一些被切掉，因此要用比較大的框架裝
    return rotated
def get_rain(degree):
        original_degree = degree
        if degree==90 or degree==85:
            degree=0
        elif degree>90:
            degree-=180
        
        tmp = degree
        
        if (tmp<0):
            tmp = -tmp
        
        c_W =  H*math.cos(math.radians(tmp))*math.sin(math.radians(tmp))
        c_H =   W*math.sin(math.radians(tmp))*math.cos(math.radians(tmp))
        
        c_W = math.floor(c_W)
        c_H = math.floor(c_H)
        final = rotate(image, degree)
        
        kernel = np.array([[-1, 2, -1]])
      
        
   
        final = cv2.cvtColor(final,cv2.COLOR_BGR2GRAY)
        #for x1 in range(H):
         #   for x2 in range(W):
          #      if(final[x1][x2]>150):final[x1][x2]=255;
           #     else:final[x1][x2]=0;
        bw = cv2.adaptiveThreshold(final, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 0)
        #final:輸入的影像
        #255:把值限制在0~255
        #cv2.ADAPTIVE_THRESH_MEAN_C:用平均的方式來決定閾值。
        #cv2.THRESH_BINARY:超過閾值就設成1
        #5:以中心周圍5x5作為比較大小
        #0:用來微調，算出的平均減去這個值
        dst_bw = rotate(bw, -degree)
        dst_bw = dst_bw[c_H:c_H+H,c_W:c_W+W]
        
        ###### here ######
        #cv2.imwrite(before_save_path+name, dst_bw)

        vertical = np.copy(bw)
        rows = vertical.shape[0]
        
        #cv2.imshow('i_1',vertical) 
        
        verticalsize = opt.kernel_size
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
        vertical = cv2.erode(vertical, verticalStructure)
        #cv2.imshow('i_2',vertical) 
        vertical = cv2.dilate(vertical, verticalStructure)
        #cv2.imshow('i_3',vertical) 
        #cv2.waitKey(0)
        
        dst = rotate(vertical, -degree)
        dst = dst[c_H:c_H+H,c_W:c_W+W]
        ###### here ######
        #cv2.imwrite(save_path+name, dst)
        return dst_bw,dst

if __name__ == '__main__':
    dataset = opt.dataset
    input_path = './dataset/'+dataset+'/input/'
    save_path = './dataset/'+dataset+'/ldgp_mul/'
    before_save_path = './dataset/'+dataset+'/bf_mul/'

    try:
        os.makedirs(save_path)
    except:
        pass

    try:
        os.makedirs(before_save_path)
    except:
        pass
    
    total_time = 0
    folder = os.listdir(input_path)
    
    for name in folder:
        start = time.process_time()
        image = cv2.imread(input_path+name)
        
        H = image.shape[0]
        W = image.shape[1]

        patch_size = opt.patch_size
        H_PatchNum = int(H/patch_size)
        W_PatchNum = int(W/patch_size)
        
        degrees = []
        iiidx = 0
        for x in range(W_PatchNum):
            for y in range(H_PatchNum):
                image_patch = image[patch_size*y:patch_size*(y+1),patch_size*x:patch_size*(x+1)]
                tmp_degree = 5*hog(image_patch, show=False, save='')
                iiidx+=1
                degrees.append(tmp_degree)
        deg_list=[]
        tb={};
        for i in range(len(degrees)):
            tb[degrees[i]]=tb.get(degrees[i], 0) + 1;
            
        for i in  range(3):
            max_cluster = max(tb, key=tb.get);
            #print(max_cluster);
            #print(tb[max_cluster]);
            tb[max_cluster]=0;
            #print("-----")
            deg_list.append(max_cluster);
      
        #degree = most_frequent(degrees)
        for i in range(3):
            tmp=str(i);
            edge,r=get_rain(deg_list[i]);
            name_i=name[:-4]+"_"+tmp+".png"
            hi=before_save_path+name_i
            #print(hi)
            #print("------")
            
            cv2.imwrite(before_save_path+name_i, edge);
            cv2.imwrite(save_path+name_i, r)
        name_1=save_path+name[:-4]+"_"+"0"+".png"
        name_2=save_path+name[:-4]+"_"+"1"+".png"
        name_3=save_path+name[:-4]+"_"+"2"+".png"
        
        img_1 = cv2.imread(name_1)
        img_2 = cv2.imread(name_2)
        img_3 = cv2.imread(name_3)
        ans=img_1+img_2+img_3
        name_ans=save_path+name[:-4]+".png"
        print(name_ans)
        cv2.imwrite(name_ans, ans)
        
        
        stop = time.process_time()  
        duration = stop - start
        print("LDGP: ", name, '/Time: ',duration ) 
        total_time += duration
        
    print("LDGP Average Time: ", total_time/len(folder))
        