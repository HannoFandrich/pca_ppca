import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
from scipy.linalg import orth
from pca_functions import pca
from pca_functions import ppca
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.cluster import KMeans
import math
import joblib

def MSE(original_data,recon_data):
    # Calculate the squared differences between corresponding data points
    squared_diff = [(np.array(og) - np.array(rec)) ** 2
                    for og, rec in zip(original_data, recon_data)]
    # Calculate the mean of the squared differences
    mse = np.mean(squared_diff)

    return mse

def vary_ppca(data, num_comp, W_init, s_init,max_iter,tol,ax,load=False):
    if len(np.asarray(W_init).shape)>2:
        if load:
            x=joblib.load('x_ppca_w.joblib')
            y=joblib.load('y_ppca_w.joblib')
        else:
            x=[]
            y=[]
            it=[]
            for i,w in enumerate(W_init):
                if i==1:
                    continue
                print(w)
                x.append(i)
                ppca_data, W, s_squared, E_xt, recon_data_ppca, iter = ppca(data, num_comp,w,s_init)
                y.append(MSE(np.asarray(data),recon_data_ppca))
                it.append(iter)
            joblib.dump(x,'x_ppca_w.joblib')
            joblib.dump(y,'y_ppca_w.joblib')
            joblib.dump(it,'it_ppca_w.joblib')
            ax.plot(x, y,label='MSE')
            ax.set_xlabel('W_init instance')
            ax.set_ylabel('Mean squared error')
            ax_n=ax.twinx()
            ax_n.plot(x,it,label='iterations')
            ax_n.set_ylabel('iterations')


    
    elif type(s_init) != int:
        if load:
            x=joblib.load('x_ppca_s.joblib')
            y=joblib.load('y_ppca_s.joblib')
        else:
            x=[]
            y=[]
            it=[]
            for i,s in enumerate(s_init):
                x.append(s)
                ppca_data, W, s_squared, E_xt, recon_data_ppca, iter = ppca(data, num_comp,W_init,s)
                y.append(MSE(np.asarray(data),recon_data_ppca))
            joblib.dump(x,'x_ppca_s.joblib')
            joblib.dump(y,'y_ppca_s.joblib')
            ax.plot(x, y,label='MSE')
            ax.set_xlabel('s_init')
            ax.set_ylabel('Mean squared error')
            ax_n=ax.twinx()
            ax_n.plot(x,it,label='iterations')
            ax_n.set_ylabel('iterations')
'''
    elif len(max_iter)>1:
        if load:
            x=joblib.load('x_ppca_iter.joblib')
            y=joblib.load('y_ppca_iter.joblib')
        else:
            x=[]
            y=[]
            for i,max_i in enumerate(max_iter):
                x.append(s)
                ppca_data, W, s_squared, E_xt, recon_data_ppca, iter = ppca(data, num_comp,W_init,s_init,max_iter=max_i)
                y.append(MSE(np.asarray(data),recon_data_ppca))
            joblib.dump(x,'x_ppca_iter.joblib')
            joblib.dump(y,'y_ppca_iter.joblib')
            ax.plot(x, y)
            ax.set_xlabel('s_init')
            ax.set_ylabel('Mean squared error')
    
    elif len(tol)>1:
        if load:
            x=joblib.load('x_ppca_tol.joblib')
            y=joblib.load('y_ppca_tol.joblib')
        else:
            x=[]
            y=[]
            for i,t in enumerate(tol):
                x.append(s)
                ppca_data, W, s_squared, E_xt, recon_data_ppca, iter = ppca(data, num_comp,W_init,s,max_iter=max_iter,tol=t)
                y.append(MSE(np.asarray(data),recon_data_ppca))
            joblib.dump(x,'x_ppca_tol.joblib')
            joblib.dump(y,'y_ppca_tol.joblib')
            ax.plot(x, y)
            ax.set_xlabel('s_init')
            ax.set_ylabel('Mean squared error')
    '''
data = pd.read_csv('highly_variable_genes.csv')
n_samples,n_features=data.shape

num_components = 2

#w_arr = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]
#W_init_arr = [np.full((n_features, num_components), a) for a in w_arr]

W_init = np.random.randn(data.shape[1], num_components)
W_init_arr=[W_init * a for a in [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]]


sa1=np.arange(0.01,0.1,0.01)
sa2=np.arange(0.1,1,0.1)
sa3=np.arange(1,100,10)
sa4=np.arange(100,1000,100)
s_init_arr=np.concatenate((sa1,sa2,sa3,sa4))

iter=np.arange(1,100,1)

t1=np.arange(0.0001,0.01,0.001)
t2=np.arange(0.01,0.1,0.01)
t3=np.arange(0.1,1,0.1)
tol=np.concatenate((t1,t2,t3))

fig = plt.figure(figsize=(8, 6))
gs = GridSpec(1,2, figure=fig)

# Define plot locations
ax1 = fig.add_subplot(gs[0,0])  # Row 0, Col 0-1  
ax2 = fig.add_subplot(gs[0,1])   # Row 1, Col 0 

W_init = np.random.randn(data.shape[1], num_components)
s_init=1
vary_ppca(data, num_components, W_init_arr, s_init,max_iter=100,tol=0.0001,ax=ax1,load=False)
vary_ppca(data, num_components, W_init, s_init_arr,max_iter=100,tol=0.0001,ax=ax2,load=False)

fig.tight_layout()
fig.savefig('fig3.png')