import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pca_functions import pca
from pca_functions import ppca
from pca_functions import RMSE
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import joblib

def vary_num_comp(data,start_comp,end_comp,ax,anno=False,load=False):
    if load:
        x=joblib.load('x_vary_num_comp.joblib')
        y1=joblib.load('y1_vary_num_comp.joblib')
        y2=joblib.load('y2_vary_num_comp.joblib')
        y3=joblib.load('y3_vary_num_comp.joblib')
    else:
        x=[]
        y1=[]
        y2=[]
        y3=[]
        for i in np.arange(start_comp,end_comp,1):
            x.append(i)

            pca_data, pca_eig_vecs, pca_eig_vals, recon_data_pca = pca(data, i)
            y1.append(RMSE(np.asarray(data),recon_data_pca))

            W_init = np.random.randn(data.shape[1], i)
            s_init=1
            ppca_data, W, s_squared, E_xt, recon_data_ppca, iter = ppca(data, i,W_init,s_init)
            y2.append(RMSE(np.asarray(data),recon_data_ppca))
            y3.append(RMSE(recon_data_pca,recon_data_ppca))
        joblib.dump(x,'x_vary_num_comp.joblib')
        joblib.dump(y1,'y1_vary_num_comp.joblib')
        joblib.dump(y2,'y2_vary_num_comp.joblib')
        joblib.dump(y3,'y3_vary_num_comp.joblib')
    ax.plot(x, y1, label="pca")
    ax.plot(x, y2, label="ppca")
    #ax.legend(loc='upper center')
    ax.set_xlabel('number of principal components')
    ax.set_ylabel('Rooted mean squared error')
    ax_n=ax.twinx()
    ax_n.plot(x,y3,color='green')
    ax_n.set_ylim(0,0.3)
    ax_n.set_ylabel('Rooted mean squared error\n pca data vs ppca data')
    ax.legend(loc='upper right')
    
    if anno!= False:
        ax.annotate(anno, xy=(0.02, 1.05), xycoords='axes fraction', fontsize=16, fontweight='bold')


def vary_ppca(data, num_comp, W_init, s_init,max_iter,tol,ax,load=False,anno=False):
    if len(np.asarray(W_init).shape)>2:
        if load:
            x=joblib.load('x_ppca_w.joblib')
            y=joblib.load('y_ppca_w.joblib')
            it=joblib.load('it_ppca_w.joblib')
        else:
            x=[]
            y=[]
            it=[]
            for i,w in enumerate(W_init):
                x.append(i)
                ppca_data, W, s_squared, E_xt, recon_data_ppca, iter = ppca(data, num_comp,w,s_init)
                y.append(RMSE(np.asarray(data),recon_data_ppca))
                it.append(iter)
            joblib.dump(x,'x_ppca_w.joblib')
            joblib.dump(y,'y_ppca_w.joblib')
            joblib.dump(it,'it_ppca_w.joblib')
        ax.plot(x, y,label='RMSE',color='blue')
        ax.set_xlabel('initial W multiplication factor')
        ax.set_ylabel('Rooted mean squared error')
        ax_n=ax.twinx()
        ax_n.plot(x,it,label='iterations',color='orange')
        ax_n.set_ylabel('iterations')
        ax.legend(loc='upper left')
        ax_n.legend(loc='upper right')

    elif type(s_init) != int:
        if load:
            x=joblib.load('x_ppca_s.joblib')
            y=joblib.load('y_ppca_s.joblib')
            it=joblib.load('it_ppca_s.joblib')
        else:
            x=[]
            y=[]
            it=[]
            for i,s in enumerate(s_init):
                x.append(s)
                ppca_data, W, s_squared, E_xt, recon_data_ppca, iter = ppca(data, num_comp,W_init,s)
                y.append(RMSE(np.asarray(data),recon_data_ppca))
                it.append(iter)
            joblib.dump(x,'x_ppca_s.joblib')
            joblib.dump(y,'y_ppca_s.joblib')
            joblib.dump(it,'it_ppca_s.joblib')
        ax.plot(x, y,label='RMSE',color='blue')
        ax.set_xlabel('initial sigma squared')
        ax.set_ylabel('Rooted mean squared error')
        ax_n=ax.twinx()
        ax_n.plot(x,it,label='iterations',color='orange')
        ax_n.set_ylabel('iterations')
        ax.legend(loc='upper left')
        ax_n.legend(loc='upper right')

    if anno!= False:
        ax.annotate(anno, xy=(0.02, 1.05), xycoords='axes fraction', fontsize=16, fontweight='bold')

data = pd.read_csv('highly_variable_genes.csv')
n_samples,n_features=data.shape

num_components = 254

W_init_arr=[np.random.randn(data.shape[1], num_components) * a for a in [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]]


sa1=np.arange(0.01,0.1,0.01)
sa2=np.arange(0.1,1,0.1)
sa3=np.arange(1,50,10)
#sa4=np.arange(100,1000,100)
#s_init_arr=np.concatenate((sa1,sa2,sa3))
s_init_arr=np.arange(0,20,0.01)

fig = plt.figure(figsize=(8, 6))
gs = GridSpec(2,2, figure=fig)

# Define plot locations
ax0 = fig.add_subplot(gs[0,:])
ax1 = fig.add_subplot(gs[1,0])  # Row 0, Col 0-1  
ax2 = fig.add_subplot(gs[1,1])   # Row 1, Col 0 

W_init = np.random.randn(data.shape[1], num_components)
s_init=1
vary_num_comp(data,2,700,ax0,anno='A',load=True)
vary_ppca(data, num_components, W_init_arr, s_init,max_iter=100,tol=0.0001,ax=ax1,load=True,anno='B')
vary_ppca(data, num_components, W_init, s_init_arr,max_iter=100,tol=0.0001,ax=ax2,load=True,anno='C')

fig.tight_layout()
fig.savefig('fig3.png')