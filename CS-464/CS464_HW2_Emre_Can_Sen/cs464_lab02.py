# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 15:31:41 2022

@author: Emre
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

def open_directory(path, size):
    im_l =[]
    im_flattened = []
    for image in Path(path).glob('*'):
        img=Image.open(image)
        img_array=np.array(img.resize((size,size), Image.BILINEAR))
        im_l.append(img_array)
        img_flat = img_array.flatten().reshape((size*size),3)
        im_flattened.append(img_flat)
    return im_l, im_flattened


def PCA(array, num_comp):
    covariance = np.cov(array)
    
    eig_val, eig_vec = np.linalg.eigh(covariance)  
    sort_eig  = np.argsort(-eig_val)
    eig_val = eig_val[sort_eig]
    eig_vec = eig_vec[:, sort_eig] 
       
    Projection = eig_vec[:, range(num_comp)] 
    Z = Projection @ Projection.T    
    return Z

file_directory = 'C:/Users/Emre/.spyder-py3/afhq_dog'

im_l, im_flattened = open_directory(file_directory, 64)

im_rgb = np.asarray(im_flattened)

im_red = im_rgb[:,:,0]
im_green = im_rgb[:,:,1]
im_blue = im_rgb[:,:,2]

# for k in (1000):
color_list = []
PVEs = []    
for color in (im_red, im_green, im_blue):
    a = PCA(color.T, 10)
    color_list.append(a)
    
Red = im_red @ color_list[0]
Green = im_green @ color_list[1]  
Blue = im_blue @ color_list[2]

pca_out = np.array([Red.T, Green.T, Blue.T])
pca_out -= pca_out.min()
pca_out /= pca_out.ptp()   
pca_out_Transpose = pca_out.T
pca_out_final = pca_out_Transpose[:10,:,:].reshape(10,64,64,3)
# pca_out_final = pca_out_Transpose[0,:,:].reshape(64,64,3)

for i in range(10):
    plt.figure(i+1)
    plt.imshow(Image.fromarray((pca_out_final[i,:,:,:]*255).astype(np.uint8)))
    plt.show()
    
# img1 =(pca_out_final*255).astype(np.uint8)
# plt.imshow(img1)

# img1 = im_l[0] @ (first * 255).astype(np.uint8)
# plt.imshow(img1)