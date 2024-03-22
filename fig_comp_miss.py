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

data = pd.read_csv('highly_variable_genes.csv')
n_samples,n_features=data.shape
#data=data.T

def vary_num_comp(data,start_comp,end_comp,ax,anno=False,load=False):
    if load:
        x=joblib.load('x_vary_num_comp.joblib')
        y1=joblib.load('y1_vary_num_comp.joblib')
        y2=joblib.load('y2_vary_num_comp.joblib')
    else:
        x=[]
        y1=[]
        y2=[]
        for i in np.arange(start_comp,end_comp,1):
            x.append(i)

            pca_data, pca_eig_vecs, pca_eig_vals, recon_data_pca = pca(data, i)
            y1.append(MSE(np.asarray(data),recon_data_pca))

            W_init = np.random.randn(data.shape[1], i)
            s_init=1
            ppca_data, W, s_squared, E_xt, recon_data_ppca, iter = ppca(data, i,W_init,s_init)
            y2.append(MSE(np.asarray(data),recon_data_ppca))
        joblib.dump(x,'x_vary_num_comp.joblib')
        joblib.dump(y1,'y1_vary_num_comp.joblib')
        joblib.dump(y2,'y2_vary_num_comp.joblib')
    ax.plot(x, y1, label="pca")
    ax.plot(x, y2, label="ppca")
    ax.legend()
    ax.set_xlabel('number of principal components')
    ax.set_ylabel('Mean squared error')
    if anno!= False:
        ax.annotate(anno, xy=(0.02, 0.90), xycoords='axes fraction', fontsize=16, fontweight='bold')


num_components = 2

def replace_with_nans(data, fraction):
    result = data.copy()  # Make a copy to avoid modifying the original array
    n_samples,n_features = data.shape
    for i in range(n_samples):
        # Choose one or two random indices in each row
        idx = np.random.choice(n_features, size=math.ceil(n_features*fraction/100), replace=False)
        # Replace the values at those indices with NaN
        result[i, idx] = np.nan
    return result


#compromised_data=replace_with_nans(np.asarray(data),10)

def test_for_mising_data(data,n_comp,ax,anno=False,load=False):
    fraction=[0.1,0.3,0.6,1,2,3,4,5,10,15,20,25,30,35,40,45,50,55,60,65,70,80,90,99]
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
            y1.append(MSE(np.asarray(data),recon_data_pca))
            y2.append(MSE(np.asarray(data),recon_data_ppca))
        joblib.dump(y1,'y1_vary_miss.joblib')
        joblib.dump(y2,'y2_vary_miss.joblib')
        print(y1)
        print(y2)
    ax.plot(fraction, y1,label='pca')
    ax.plot(fraction, y2,label='ppca')
    ax.set_xlabel('fraction of observables missing per row')
    ax.set_ylabel('Mean squared error')
    ax.legend()
    if anno!= False:
        ax.annotate(anno, xy=(0.02, 0.90), xycoords='axes fraction', fontsize=16, fontweight='bold')

fig = plt.figure(figsize=(8, 6))
gs = GridSpec(2,1, figure=fig)

# Define plot locations
ax1 = fig.add_subplot(gs[0])  # Row 0, Col 0-1  
ax2 = fig.add_subplot(gs[1])   # Row 1, Col 0 


print(np.argwhere(np.isnan(np.asarray(data))))
vary_num_comp(data,2,700,ax1,anno='A',load=False)
#w_arr = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]
#W_init_arr = [np.full((n_features, num_components), a) for a in w_arr]
#vary_ppca(data, num_components, W_init_arr, s_init=1,ax=ax2)


test_for_mising_data(data,num_components,ax2,anno='B',load=False)
fig.tight_layout()
fig.savefig('fig2.png')


