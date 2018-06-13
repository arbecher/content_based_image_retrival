import numpy as np
import imageio
import matplotlib.pyplot as plt
import random
from scipy.spatial import distance
from stl10_input import *

option = 2  #Option=1 para ajuste gamma e option=2 para equalização de histogramas
list_all_images = read_all_images("unlabeled_X.bin")
g = 1.2    #para ajuste gamma
batch = 100
#Luminance (único canal de cor)
images = (0.299*list_all_images[0:batch, :, :, 0] + 0.587*list_all_images[0:batch, :, :, 1] + 0.114*list_all_images[0:batch, :, :, 2]).astype(np.uint8)
image_test = (0.299*list_all_images[1000, :, :, 0] + 0.587*list_all_images[1000, :, :, 1] + 0.114*list_all_images[1000, :, :, 2]).astype(np.uint8)


######################  FUNÇÕES  ######################################################
#######################################################################################

#Função de ajuste gamma
def gamma_ajust(img, g):
    p = float(1/g)
    images_eq = img**(p)
    return(images_eq)

# Função de histograma de cores
def hist(M):
    m, n = M.shape
    hist_color = np.zeros(256).astype(int)
    for i in range(m):
        for j in range(n):
            hist_color[M[i, j]]+=1
    return(hist_color)

def hist_cumulative(h):
    ha=([0]*256)
    ha[0]=h[0]
    for i in range(1,256):
        ha[i]=ha[i-1]+h[i]
    return(ha)

# Função densidade
def density(hist):
    d_c = hist/(np.sum(hist))
    return(d_c)

##Função de equalização
def equalization(img, ha):
    m, n = img.shape
    image = (np.zeros_like(img)).astype(np.uint8)
    ha = np.array(ha)/(m*n)
    t = np.uint8(255*ha)
    for i in range(m):
        for j in range(n):
            image[i, j] = t[img[i, j]]
    return(image)


###-----------------K-MEANS-----------------------------

def kmeans(dataset, centers, threshold):

    ids = np.sort(random.sample(range(0,batch), centers))
    centroids = dataset[ids, :]
    divergence = 2*threshold
    distances = np.zeros((batch, centers)).astype(float)
    while divergence > threshold:
        for i in range(centers):
            for j in range(batch):
                distances[j, i] = distance.euclidean(centroids[i],dataset[j])
        clusters = np.zeros(batch).astype(int)
        for i in range(batch):
            ind = np.where(distances[i] == np.min(distances[i]))
            clusters[i] = ind[0]

        divergence = 0
        for i in range(centers):
            index = np.where(clusters == i)
            old_centroid = centroids[i, :]
            centroids[i, :] = np.mean(dataset[index, :], axis=1)
            divergence = (divergence + distance.euclidean(centroids[i, :],old_centroid))/256

    return(clusters, centroids, centers)


########################################################################################
########################################################################################

############# Pré processamento
#Usa ajuste gamma
if option == 1:
    image_test = gamma_ajust(image_test)
    for i in range(batch):
         images[i] = gamma_ajust(images[i], g)
#usa equalização de histogramas
if option == 2:
    h_it = hist(image_test)
    ha_it = hist_cumulative(h_it)
    image_test = equalization(image_test, ha_it)
    H = np.zeros((batch, 256)).astype(float)
    for i in range(batch):
        H[i] = hist(images[i])
        H[i] = hist_cumulative(H[i])
        images[i] = equalization(images[i], H[i])


#####---------- Extração de características-----------

H_color = np.zeros((batch, 256)).astype(int)
d_c = np.zeros((batch, 256)).astype(float)
for i in range(batch):
    H_color[i] = hist(images[i])
    d_c[i] = density(H_color[i])
dc_it = density(ha_it)

clusters, centroids, centers = kmeans(d_c, 50, 1e-7)
dist_min = 1000
for i in range(centers):
     closest = distance.euclidean(centroids[i, :],dc_it)
     if closest < dist_min:
         dist_min = closest
         ind = i


indices = np.where(clusters == ind)

return_images= images[indices, :]


#
#plt.imshow(images[1], cmap='gray')
#plt.subplot(122)
#plt.imshow(images[20], cmap='gray')
#plt.subplot(212)
#plt.imshow(images[40], cmap='gray')
#plt.subplot(224)
#plt.imshow(list_all_images[1000], cmap='gray')
#plt.show()
