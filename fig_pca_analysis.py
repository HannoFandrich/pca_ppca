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


def scoreplot(pca_data,ax,axis_lables=False,anno=False):
    data = pca_data.copy()

    num_clusters = 5  
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data)
    labels = kmeans.labels_

    for i in range(num_clusters):
        cluster_points = np.asarray(data[labels == i])
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1],label=f'Cl. {i}')

    if axis_lables:
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')
    ax.set_title('Scoreplot')
    #ax.legend(ncol=3)
    if anno!= False:
        ax.annotate(anno, xy=(0.02, 0.90), xycoords='axes fraction', fontsize=16, fontweight='bold')



def explained_variance_plot(eig_vals, ax, label,anno=False):
      total_var = np.sum(eig_vals)
      explained_var = np.asarray(
      [100*(i/total_var) for i in sorted(eig_vals, reverse=True)])
      ax.set_title("Scree plots")
      cumulative_explained_var=np.cumsum(explained_var)
      for i,c in enumerate(cumulative_explained_var):
          if c > 60:
              min_comp=i
              break
      ax.bar(x=np.arange(1,np.shape(explained_var)[0]+1), 
            height=cumulative_explained_var, 
            width=0.4, color="green",label="cumul. explained variance ")
      ax.plot(np.arange(1,np.shape(explained_var)[0]+1), 
            explained_var, 
            linestyle="--",marker='o',color='red', markersize=15, label="explained variance ")
      ax.legend()
      ax.set_xlim(0,10.5)
      ax.set_ylim(0,30)
      if anno!= False:
        ax.annotate(anno, xy=(0.02, 0.90), xycoords='axes fraction', fontsize=16, fontweight='bold')
      return explained_var,min_comp


fig = plt.figure(figsize=(8, 6))
gs = GridSpec(2,2, figure=fig)

ax1 = fig.add_subplot(gs[0, :])  # Row 0, Col 0-1  
ax2 = fig.add_subplot(gs[1, 0])   # Row 1-2, Col 0
ax3 = fig.add_subplot(gs[1, 1])  # Row 1-2, Col 1-2



num_components = len(list(data.columns))-100

pca_data, eig_vecs_selected, eig_vals_selected, recon_data_pca = pca(data, num_components)
W_init = np.random.randn(data.shape[1], num_components)
s_init=1
ppca_data, W, s_squared, E_xt, recon_data_ppca, iter = ppca(data, num_components,W_init,s_init)

explained_var, min_comp=explained_variance_plot(eig_vals_selected, ax1, 'pca',anno='A')
print(min_comp)

num_components = 2
pca_data, pca_eig_vecs, pca_eig_vals, recon_data_pca = pca(data, num_components)
W_init = np.random.randn(data.shape[1], num_components)
s_init=1
ppca_data, W, s_squared,  E_xt, recon_data_ppca, iter = ppca(data, num_components,W_init,s_init)

scoreplot(pca_data,ax2, axis_lables=True,anno='B')
scoreplot(ppca_data,ax3,anno='C')

fig.tight_layout()
fig.savefig('fig1.png')


fig = plt.figure(figsize=(8, 6))
gs = GridSpec(2,2, figure=fig)

ax1 = fig.add_subplot(gs[0, 0])  # Row 0, Col 0-1  
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])   # Row 1-2, Col 0
ax4 = fig.add_subplot(gs[1, 1])  # Row 1-2, Col 1-2

letters=['A','B','C','D']
for i,a in enumerate([ax1,ax2,ax3,ax4]):
    W_init = np.random.randn(data.shape[1], num_components)
    s_init=1
    ppca_data, W, s_squared,  E_xt, recon_data_ppca, iter = ppca(data, num_components,W_init,s_init)
    scoreplot(ppca_data,a, axis_lables=False,anno=letters[i])

fig.tight_layout()
fig.savefig('fig_a1.png')


