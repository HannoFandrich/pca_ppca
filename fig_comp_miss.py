import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
from scipy.linalg import orth
from pca_functions import pca
from pca_functions import ppca
from pca_functions import RMSE
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.cluster import KMeans
import math
import joblib

data = pd.read_csv('highly_variable_genes.csv')
n_samples,n_features=data.shape
#data=data.T

def replace_with_nans(data, fraction):
    result = data.copy()  # Make a copy to avoid modifying the original array
    n_samples,n_features = data.shape
    for i in range(n_samples):
        # Choose one or two random indices in each row
        idx = np.random.choice(n_features, size=math.ceil(n_features*fraction/100), replace=False)
        # Replace the values at those indices with NaN
        result[i, idx] = np.nan
    return result

def test_for_mising_data(data,n_comp,ax,anno=False,load=False):
    #fraction=[0.1,0.3,0.6,1,2,3,4,5,10,15,20,25,30,35,40,45,50,55,60,65,70,80,90,99]
    fraction=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    
    if load:
        y1=joblib.load('y1_vary_miss.joblib')
        y2=joblib.load('y2_vary_miss.joblib')
    else:
        y1=[]
        y2=[]
        for f in fraction:
            compromised_data=replace_with_nans(np.asarray(data),f)
            pca_data, pca_eig_vecs, pca_eig_vals, recon_data_pca=pca(compromised_data,n_comp,)
            W_init = np.random.randn(data.shape[1], n_comp)
            s_init=1
            ppca_data, W, s_squared, E_xt, recon_data_ppca, iter = ppca(compromised_data, n_comp,W_init,s_init)
            y1.append(RMSE(np.asarray(data),recon_data_pca))
            y2.append(RMSE(np.asarray(data),recon_data_ppca))
        joblib.dump(y1,'y1_vary_miss.joblib')
        joblib.dump(y2,'y2_vary_miss.joblib')
    ax.plot(fraction, y1,label='pca')
    ax.plot(fraction, y2,label='ppca')
    ax.set_xlabel('fraction of observables missing per row')
    ax.set_ylabel('Rooted mean squared error')
    ax.legend(loc='upper center')
    if anno!= False:
        ax.annotate(anno, xy=(0.02, 0.90), xycoords='axes fraction', fontsize=16, fontweight='bold')

def test_for_mising_data_vary_comp(data,ax,anno=False,load=False):
    #fraction=[0.1,0.3,0.6,1,2,3,4,5,10,15,20,25,30,35,40,45,50,55,60,65,70,80,90,99]
    num_comp=np.arange(2,254,20)
    fraction=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    
    if load:
        y1=joblib.load('y1_vary_miss.joblib')
    else:
        y1=[]
        for f in fraction:
            compromised_data=replace_with_nans(np.asarray(data),f)
            pca_data, pca_eig_vecs, pca_eig_vals, recon_data_pca=pca(compromised_data,254)
            y1.append(RMSE(np.asarray(data),recon_data_pca))
        joblib.dump(y1,'y1_vary_miss.joblib')
    #ax.plot(fraction, y1,label='pca n.c.:254')

    if load:
        yy2=joblib.load('yy2_vary_miss.joblib')
    else:
        yy2=[]
        for n in num_comp:

            y2=[]
            for f in fraction:
                compromised_data=replace_with_nans(np.asarray(data),f)
                W_init = np.random.randn(data.shape[1], n)
                s_init=1
                ppca_data, W, s_squared, E_xt, recon_data_ppca, iter = ppca(compromised_data, n,W_init,s_init)
                y2.append(RMSE(np.asarray(data),recon_data_ppca))
            yy2.append(y2)
        joblib.dump(yy2,'yy2_vary_miss.joblib')
    for i,y in enumerate(yy2):
        ax.plot(fraction, y,label='pca n.c.:'+str(num_comp[i]))

    ax.set_xlabel('fraction of observables missing per row')
    ax.set_ylabel('Rooted mean squared error')
    ax.legend(loc='upper center')
    if anno!= False:
        ax.annotate(anno, xy=(0.02, 0.90), xycoords='axes fraction', fontsize=16, fontweight='bold')



fig = plt.figure(figsize=(8, 7))
gs = GridSpec(2,1, figure=fig)

# Define plot locations
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])  # Row 0, Col 0-1  


print(np.argwhere(np.isnan(np.asarray(data))))

num_components = 2
test_for_mising_data(data,num_components,ax1,anno='A',load=False)
test_for_mising_data_vary_comp(data,ax2,anno='B',load=True)
fig.tight_layout()
fig.savefig('fig2.png')


