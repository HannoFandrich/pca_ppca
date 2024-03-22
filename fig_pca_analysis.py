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


data = pd.read_csv('highly_variable_genes.csv')
#data=data.T

num_components = 2

pca_data, pca_eig_vecs, pca_eig_vals, recon_data_pca = pca(data, num_components)
W_init = np.random.randn(data.shape[1], num_components)
s_init=1
ppca_data, W, s_squared, E_xt, recon_data_ppca, iter = ppca(data, num_components,W_init,s_init)

reconstructed_data_pca = np.dot(pca_data, pca_eig_vecs.T) + np.mean(np.asarray(data), axis=0)

fig = plt.figure(figsize=(8, 6))
gs = GridSpec(2,2, figure=fig)

# Define plot locations
ax1 = fig.add_subplot(gs[0, :])  # Row 0, Col 0-1  
ax2 = fig.add_subplot(gs[1, 0])   # Row 1-2, Col 0
ax3 = fig.add_subplot(gs[1, 1])  # Row 1-2, Col 1-2


def scoreplot(pca_data,ax,axis_lables=False,anno=False):
    X_r = pca_data

    # Perform KMeans clustering
    num_clusters = 5  # Number of clusters you want
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(X_r)
    labels = kmeans.labels_

    for i in range(num_clusters):
        cluster_points = X_r[labels == i]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1],label=f'Cl. {i}')

    if axis_lables:
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')
    ax.set_title('Scoreplot')
    #ax.legend(ncol=3)
    ax.grid(True)
    if anno!= False:
        ax.annotate(anno, xy=(0.02, 0.90), xycoords='axes fraction', fontsize=16, fontweight='bold')

def loadingplot(data,pca_eig_vecs,ax,features):
    # Add variable loadings
    scaling=1
    for i, feature in enumerate(features):
        ax.arrow(0, 0, pca_eig_vecs[i,0]*scaling, pca_eig_vecs[i,1]*scaling,
                head_width=0.0001, head_length=0.0001, color='k')
        ax.text(pca_eig_vecs[i,0] * 1.15*scaling, pca_eig_vecs[i,1] * 1.15*scaling,
                feature, color='k')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Loadingplot')
    ax.legend()
    ax.grid(True)

def filter_features(pca_eig_vecs):
    indices = []
    max_length = max([pca_eig_vecs[i,0] + pca_eig_vecs[i,1] for i in range(len(pca_eig_vecs))])
    for i in range(len(pca_eig_vecs)):
        length = pca_eig_vecs[i,0] + pca_eig_vecs[i,1]
        if length > max_length * 0.5:
            indices.append(i)
    return indices


def explained_variance_plot(eig_vals, ax, label,anno=False):
      total_var = np.sum(eig_vals)
      explained_var = np.asarray(
      [100*(i/total_var) for i in sorted(eig_vals, reverse=True)])
      ax.set_title("Scree plots")
      cumulative_explained_var=np.cumsum(explained_var)
      ax.bar(x=np.arange(2,np.shape(explained_var)[0]+2), 
            height=cumulative_explained_var, 
            width=0.4, color="green",label="cumul. explained variance ")
      ax.plot(np.arange(2,np.shape(explained_var)[0]+2), 
            explained_var, 
            linestyle="--",marker='o',color='red', markersize=15, label="explained variance ")
      ax.legend()
      ax.grid()
      ax.set_xlim(1,10.5)
      ax.set_ylim(0,30)
      if anno!= False:
        ax.annotate(anno, xy=(0.02, 0.90), xycoords='axes fraction', fontsize=16, fontweight='bold')
      return explained_var

num_components = len(list(data.columns))-100

pca_data, pca_eig_vecs, pca_eig_vals = pca(data, num_components)
W_init = np.random.randn(data.shape[1], num_components)
s_init=1
ppca_data, W, s_squared, E_xt, recon_data_ppca, iter = ppca(data, num_components,W_init,s_init)

#reconstructed_data_pca = np.dot(pca_data, pca_eig_vecs.T) + np.mean(np.asarray(data), axis=0)
#reconstructed_data_ppca = np.dot(ppca_data, W.T) + np.mean(np.asarray(data), axis=0)

explained_var=explained_variance_plot(pca_eig_vals, ax1, 'pca',anno='A')
#explained_variance_plot(ppca_eig_vals, ax1, 'ppca')

num_components = 2
pca_data, pca_eig_vecs, pca_eig_vals, recon_data_pca = pca(data, num_components)
W_init = np.random.randn(data.shape[1], num_components)
s_init=1
ppca_data, W, s_squared,  E_xt, recon_data_ppca, iter = ppca(data, num_components,W_init,s_init)

filtered_features= [list(data.columns)[i] for i in filter_features(pca_eig_vecs)]
#loadingplot(data,pca_eig_vecs,ax2,filtered_features)
scoreplot(pca_data,ax2, axis_lables=True,anno='B')
scoreplot(E_xt,ax3,anno='C')
#loadingplot(data,W,ax3,filtered_features)


fig.tight_layout()
fig.savefig('fig1.png')