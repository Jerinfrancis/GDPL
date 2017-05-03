# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 13:37:04 2017

@author: SFRPL-2
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 15:11:04 2017

@author: Jerin Francis
"""
from matplotlib import pyplot as plt
from scipy import misc
from scipy.stats import mode
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.signal import order_filter

def sharpen_foreground(img,bg):
  for col in range(np.shape(img)[1]):
    column=img[:,col]
    fg=column!=bg
    idx=np.where(np.diff(fg))[0]+1
    start = idx[::2]
    end = idx[1::2]
    for i in range(len(end)):
      img[start[i]:end[i],col]=mode(column[start[i]:end[i]])[0]
    #img=remove_small_objects(g.astype('uint'), min_size=5,connectivity=1)
  return img
plt.rcParams['image.cmap'] = 'hot'
    
data = misc.imread('D:\Data\GDPL\OP & PV\loop_2_PV_SP.jpg')[:,:,:3]
ncluster=input('Enter No of Distinct colors :')
#ncluster=6

y,x,z = np.shape(data)
plt.imshow(data)

df = pd.DataFrame(data.reshape(y*x,z))
bg_val = df.mode().values[0]
rgb = np.reshape(df.values.astype('float'),[y,x,z])
r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]

sharp=np.zeros_like(data)
sharp[:,:,0]=sharpen_foreground(r,bg_val[0])
sharp[:,:,1]=sharpen_foreground(g,bg_val[1])
sharp[:,:,2]=sharpen_foreground(b,bg_val[2])
sp = pd.DataFrame(sharp.reshape(y*x,z))
kmeans = KMeans(init='random', n_clusters=ncluster, n_init=20)
kmeans.fit(sp)
kmeans.cluster_centers_
dft=kmeans.predict(sp)
cimg = np.reshape(dft,[y,x])
curves = np.zeros([ncluster,x])
domain=np.ones([5,5])
for cv in range(ncluster):
  #plt.figure()
  #plt.imshow(cimg==cv)
  #cvn = order_filter(cimg==cv,domain,20).astype('bool')
  cvn = cimg==cv
  for i in range(x):
    curves[cv,i]=y-np.median(np.where(cvn[:,i]))
plt.figure()
for i in range(0,ncluster):
  plt.scatter(np.linspace(0,1,x),curves[i],s=5)
  plt.legend(loc=i)
  plt.ylim([0,y])
