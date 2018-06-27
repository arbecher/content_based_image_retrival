# Processamento de Imagens - 2018/01
# Aline Regina Becher - NUSP: 10492388

#Código Fonte



import numpy as np
import imageio
import matplotlib.pyplot as plt
import random
from stl10_input import *
from functions_a import *
import cv2
import time

inicio = time.time()
b = 5
nbins = 2**b
t = np.random.randint(5000)
list_all_images = read_all_images("test_X.bin")
image_search = read_all_images("train_X.bin")[t]
bshift = np.uint8(8 - b)
print(t)
#------------------ Código ---------------------------------------

### equalização de histograma image_search ---------------------------------
image_search2 = hsv(image_search)
hist_color = hist(image_search2[:, :, 2], 256)
ha = hist_cumulative(hist_color, 256)
image_search2[:, :, 2] = equalization(image_search2[:, :, 2], ha)
image_search = cv2.cvtColor(image_search2, cv2.COLOR_HSV2BGR)


### equalização de histograma image_search ----------------------------
Images = (np.zeros_like(list_all_images)).astype(np.uint8)
for i in range(8000):
    list_all_images[i, :, :, :] = hsv(list_all_images[i, :, :, :])
    hist_color = hist(list_all_images[i, :, :, 2], 256)
    ha = hist_cumulative(hist_color, 256)
    list_all_images[i, :, :, 2] = equalization(list_all_images[i, :, :, 2], ha)
    Images[i, :, :, :] = cv2.cvtColor(list_all_images[i, :, :, :], cv2.COLOR_HSV2BGR)


#-------------------- vetor de características image_Search----------

image_search = image_search.astype(np.uint8)
image_search = (image_search >> bshift)
hist_color_0 = hist(image_search[:, :, 0],  nbins)
hist_color_1 = hist(image_search[:, :, 1], nbins)
hist_color_2 = hist(image_search[:, :, 2], nbins)
dc_image_search = np.concatenate((hist_color_0, hist_color_1))
dc_image_search = np.concatenate((dc_image_search, hist_color_2))

#-------------------------------------------------------------------
#---------------vetor de características Imagens------------------------

Images = Images.astype(np.uint8)
Images = (Images >> bshift)
D = np.zeros((8000, 3*nbins)).astype(float)
for i in range(8000):
    d = []
    for j in range(3):
        hist_c = hist(Images[i, :, :, j], nbins)
        d = np.concatenate((d, hist_c), nbins)
    D[i, :] = d

clusters, centroids, centers, ids = kmeans(D, 600, 1e-7, 1, 1)
dist_min = 1000000000
for i in range(centers):
     closest = hist_correl(centroids[i, :], dc_image_search)
     if closest < dist_min:
         dist_min = closest
         ind = i

ids1 = np.where(clusters == ind)
ids1 = ids1[0]
print(ids1)

#-------------------Para textura ---------------------------------
image_search_ =  0.299*image_search[:, :, 0] + 0.587*image_search[:, :, 1] + 0.114*image_search[:, :, 2]
image_search_ = image_search.astype(np.uint8)
image_search_ = (image_search >> bshift)
Images_ = 0.299*Images[:, :, :, 0] + 0.587*Images[:, :, :, 1] + 0.114*Images[:, :, :, 2]
bshift = np.uint8(8 - b)
Images_ = Images_.astype(np.uint8)
Images = (Images >> bshift)


#---------------------descritores de textura para image_search---------

G_is = texture_descriptors(image_search_, nbins)
u_i, u_j = medias(G_is, nbins)
sigma_i, sigma_j = desvios(G_is, u_i, u_j, nbins)
d_is = haralick_descriptors(G_is, u_i, u_j, sigma_i, sigma_j, nbins)

#------------------------------------------------------------------------

# Descritores de textura para run
#conjunto de imagens-----------
d = np.zeros((8000, 5)).astype(float)
for i in range(8000):
    G = texture_descriptors(Images_[i, :, :], nbins)
    u_i, u_j = medias(G, nbins)
    sigma_i, sigma_j = desvios(G, u_i, u_j, nbins)
    d[i, :] = haralick_descriptors(G, u_i, u_j, sigma_i, sigma_j, nbins)

clusters, centroids, centers, ids = kmeans(d, 600, 1e-7, 2, ids)
dist_min = 1000000000
for i in range(centers):
     closest = euclidean(centroids[i, :],d_is)
     if closest < dist_min:
         dist_min = closest
         ind = i

ids2 = np.where(clusters == ind)
ids2 = ids2[0]
print(ids2)
fim = time.time()
print(fim - inicio)
