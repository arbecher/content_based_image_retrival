# Processamento de Imagens - 2018/1
# Aline Regina Becher - NUSP: 10492388

# Conjunto de funções necessárias para a busca visual baseada em cor e textura.

import numpy as np
import imageio
import matplotlib.pyplot as plt
import random
import cv2




#-------------------Funções -------------------


#--------------------- Função de histograma de cores
def hist(M, bins):
    m, n = M.shape
    histcolor = np.zeros(bins).astype(int)
    for i in range(m):
        for j in range(n):
            histcolor[M[i, j]]+=1
    return histcolor

def hist_cumulative(h, bins):
    ha=([0]*bins)
    ha[0]=h[0]
    for i in range(1,bins):
        ha[i]=ha[i-1]+h[i]
    return(ha)


##--------------------------- equalização de histogramas
def equalization(img, ha):
    m, n = img.shape
    image = (np.zeros_like(img)).astype(np.uint8)
    ha = np.array(ha)/(m*n)
    t = np.uint8(255*ha)
    for i in range(m):
        for j in range(n):
            image[i, j] = t[img[i, j]]
    return(image)

#----------------------RGB ==> HSV----------------------
def hsv(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return(hsv_img)
#--------------------HSV ==> RGB-----------------------------------
def rgb(img):
    rgb = cv2.cvtColor(image_search2, cv2.COLOR_HSV2BGR)
    return(rgb_img)
#----------------------------------------------------------------


#-------------------- Correlação de histogramas - distância
def hist_correl(h1, h2):
    n = h1.shape
    h1_ = np.sum(h1)/n
    h2_ = np.sum(h2)/n
    d = (np.sum((h1 - h1_)*(h2-h2_)))/(np.sqrt(np.sum((h1 - h1_)**2))*np.sum((h2 - h2_)**2) + 1)
    return(d)
#----------------------------------------------------

#-----------------------distância euclidiana--------------
def euclidean(u, v):
    distance = np.sqrt((np.sum((u-v)**2)))
    return(distance)
#-------------------------------------------------

#----------Matriz de co-ocorrência G-------------------
def texture_descriptors(M, nbins):
    A = np.zeros((nbins, nbins))
    for i in range(96):
        for j in range(96-1):
            A[M[i, j], M[i, j+1]] +=1
    G = A/np.sum(A)
    return(G)
#--------------------------------------------------------


#--------Médias e desvios nas direções  das linhas e colunas--------
def medias(G, nbins):
    u_i = np.zeros(nbins).astype(float)
    for i in range(nbins):
        u_i[i] = i*np.sum(G[i, :])
    u_i = np.sum(u_i)

    u_j = np.zeros(nbins).astype(float)
    for j in range(nbins):
        u_j[j] = j*np.sum(G[:, j])
    u_j = np.sum(u_j)
    return(u_i, u_j)

def desvios(G, u_i, u_j, nbins):
    sigma_i = np.zeros(nbins).astype(float)
    sigma_j = np.zeros(nbins).astype(float)
    for i in range(nbins):
        sigma_i[i] = ((i-u_i)**2)*(np.sum(G[i, :]))
        sigma_j[i] = ((i-u_j)**2)*(np.sum(G[:, i]))
    sigma_i = np.sum(sigma_i)
    sigma_j = np.sum(sigma_j)
    return(sigma_i, sigma_j)
#-------------------------------------------------------------


#-----------------------Descritores de textura ----------------
def haralick_descriptors(G, u_i, u_j, sigma_i, sigma_j, nbins):
    d_t = np.zeros(5).astype(float)

    energy = np.sum(G**2)
    entropy = -np.sum(G*(np.log(G + 0.001)))

    G_ = G
    Gc = G
    Gh = G
    for i in range(nbins):
        for j in range(nbins):
            G_[i, j] = ((i-j)**2)*G[i][j]
            Gc[i, j] = i*j*G[i, j] - u_i*u_j
            Gh[i, j] = (G[i, j])/(1 + np.abs(i-j))
    contrast = (1/((nbins - 1)**2))*np.sum(G_)
    if (sigma_i*sigma_j > 0):
        correlation = np.sum(Gc)/(sigma_i*sigma_j)
    else:
        correlation = 0

    homogeneity = np.sum(Gh)

    d_t[0] = energy
    d_t[1] = entropy
    d_t[2] =contrast
    d_t[3] = correlation
    d_t[4] = homogeneity
    return(d_t)
#-----------------------------------------------------

#------------------------- K-means ----------------------
def kmeans(dataset, centers, threshold, opt, ind):
    r, s = dataset.shape
    if (opt == 1):
        ids = np.sort(random.sample(range(0,8000), centers))
    if (opt == 2):
        ids = ind
    centroids = dataset[ids, :]
    divergence = 2*threshold
    distances = np.zeros((8000, centers)).astype(float)
    while divergence > threshold:
        for i in range(centers):
            for j in range(8000):
                if (opt == 1):
                    distances[j, i] = hist_correl(centroids[i], dataset[j])
                elif (opt == 2):
                    distances[j, i] = euclidean(centroids[i],dataset[j])
        clusters = np.zeros(8000).astype(int)
        for i in range(8000):
            ind = np.where(distances[i] == np.min(distances[i]))

            clusters[i] = ind[0]
        divergence = 0
        for i in range(centers):
            index = np.where(clusters == i)
            old_centroid = centroids[i, :]
            centroids[i, :] = np.mean(dataset[index, :], axis=1)
            divergence = (divergence + euclidean(centroids[i, :],old_centroid))/s

    return(clusters, centroids, centers, ids)
#-----------------------------------------------------
