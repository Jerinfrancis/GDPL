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
from scipy.interpolate import interp1d

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
plt.style.use('ggplot')
plt.rcParams['image.cmap'] = 'hot'
plt.rcParams["figure.figsize"] = (16,9)
plt.close('all')
plt.ion()
loop_no = '10'
path = 'D:\Data\GDPL\\'
data = misc.imread(path+'OP & PV\loop_'+loop_no+'_OP.jpg')[:,:,:3]
loopdata=pd.read_csv(path+'OP & PV\loop_'+loop_no+'.csv')
plt.figure()
plt.axis("off")
plt.imshow(data)
plt.pause(0.05)
plt.draw() 
ncluster=2#input('Enter No of Distinct colors :')
#ncluster=6

y,x,z = np.shape(data)

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

for i in range(0,ncluster):
  plt.figure()
  plt.plot(np.linspace(0,1,x),curves[i])
  plt.legend(loc=i)
  plt.title('Signal :'+str(i))
  plt.ylim([0,y])
  plt.pause(0.05)
plt.draw() 
  
sig1 = loopdata['loop_'+loop_no+'.OP'].values
min_sig1 = min(sig1)
max_sig1 = max(sig1)
nm = input('Extracted signal are displayed. \nEnter number of the Signal of interest :')
exsig = curves[nm,:]
# Interpolate nans using the nearest values
nans, idx= np.isnan(exsig), lambda z: z.nonzero()[0]
exsig[nans]= np.interp(idx(nans), idx(~nans), exsig[~nans])
# Scale the extracted signal to be in the same range as the original signal
min_exsig = min(exsig)
max_exsig = max(exsig)
sig2 = (exsig-min_exsig)/(max_exsig-min_exsig)*(max_sig1-min_sig1)+min_sig1
# Interpolate the extracted signal to be of the same length as the original signal
x_large = np.linspace(0, 1, num=len(sig1), endpoint=True)
x_small = np.linspace(0, 1, num=len(sig2), endpoint=True)
f = interp1d(x_small, sig2)
sig2intpl = f(x_large)
# Compute RMSE as percentage of the mean of the signal
RMSE = np.sqrt(np.mean((sig1-sig2intpl)**2))
mean_sig1 = np.mean(sig1)
RMSEpc = RMSE/mean_sig1*100
print 'RMSE(%)  =',RMSEpc
compare = pd.DataFrame([sig1,sig2intpl]).T
compare.columns = ['Original Signal','Extracted Signal']
compare.plot()
plt.title('Loop '+loop_no+' OP Original vs Extracted. RMSE(%) = '+str(np.round(RMSEpc,3)))
compare.to_csv(path+'RMSE\Loop '+loop_no+'.csv',index=False)
plt.savefig(path+'RMSE\Loop '+loop_no+'.png')
