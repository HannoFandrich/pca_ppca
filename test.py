import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.linalg import orth
'''
data = pd.read_csv('pizza.csv')
#print(data.head())

data=data.drop(['brand','id'], axis=1)
#data=data.values

# probabilistic principal component analysis function

def ppca(t, num_comp,W_init,sigma_init, max_iter=1000, tol=1e-5):

    n_samples, n_features = t.shape

    t_norm = np.asarray(t - np.mean(np.asarray(t),axis=0)) #normalize the data
    S=np.cov(t_norm.T)

    W=W_init
    s_squared=sigma_init
    #Z = np.dot(data_norm, W) #dot product of data and W


    X=np.asarray(t)
    mu=np.mean(X.flatten())
    def e_step(X, mu, W , sigma_sq, latent_dim):
        N, M = X.shape
        M_matrix = np.dot(X, W)
        C = np.dot(W.T, W) + sigma_sq * np.eye(latent_dim)
        C_inv = np.linalg.inv(C)
        Z_mean = np.dot(np.dot(M_matrix, C_inv), (X - mu).T).T + mu
        return Z_mean

    def m_step(X, mu, W , sigma_sq, latent_dim, Z):
        N, M = X.shape
        d = latent_dim
        W_new = np.dot(np.dot(np.dot((X - mu).T, Z), np.linalg.inv(np.dot(Z.T, Z) + N * sigma_sq * np.eye(d))), np.linalg.inv(np.dot(Z.T, Z) + N * sigma_sq * np.eye(d)))
        W_new = np.dot(np.dot(np.dot((X - mu).T, Z), np.linalg.inv(np.dot(Z.T, Z) + N * sigma_sq * np.eye(d))), np.linalg.inv(np.dot(Z.T, Z) + N * sigma_sq * np.eye(d)))
        sigma_sq_new = (np.sum(np.sum((X - mu) ** 2)) - 2 * np.sum(np.sum(np.dot(Z, W_new.T) * (X - mu))) + np.sum(np.sum(np.dot(np.dot(Z, W_new.T), W_new) * Z)) + N * np.sum(np.diag(np.dot(W_new.T, W_new)))) / (N * M)
        return W_new, sigma_sq_new

    sigma=sigma_init
    E_Y = t_norm
    for i in range(max_iter):
        
        M_inv = np.linalg.inv(W.T @ W + s_squared * np.eye(num_comp))
        
        E_x = E_Y @ W @ M_inv

        s_squared_new = 1 / (n_samples * n_features) * (n_samples * sigma * np.trace(W @ M_inv @ W.T) + np.sum((E_x @ W.T - E_Y)**2) ) 

        weighted_1 = (E_Y.T @ E_x) / s_squared_new
        weighted_2 = (n_samples*sigma * M_inv + E_x.T @ E_x) / s_squared_new
        W_new = weighted_1 @ np.linalg.inv(weighted_2)
        

        M = (W.T @ W) + s_squared * np.eye(num_comp)
        
        MWSW = np.linalg.inv(M) @ W.T @ S @ W 
        W_new = S @ W @ np.linalg.inv(s_squared * np.eye(num_comp)+MWSW)
        
        SWMW = S @ W @ np.linalg.inv(M) @ W_new.T
        s_squared_new = np.trace(S - SWMW)/n_features

        # Check for convergence
        print(s_squared_new-s_squared)
        if (np.abs(s_squared_new-s_squared) < tol):
            break

        W=W_new
        s_squared=s_squared_new

    #print(W)
    basis = orth(W) # Orthonormal basis of the principal subspace
    #print(basis)
    data_p = E_Y @ basis # Projected observation matrix on the principal subspace
    S_p = np.cov(data_p.T) # Projected covariance matrix
    vals, vecs = np.linalg.eigh(S_p) 
    idx = list(reversed(np.argsort(vals))) # Sort in decreasing order
    vals = vals[idx]; vecs = vecs[:,idx]
    X = E_Y @ basis @ vecs
    W = basis @ vecs # Corrected principal components


    ppca_data = np.dot(t_norm, W)
    return ppca_data, W, s_squared, S

num_components = 2

W_init = np.random.randn(data.shape[1], num_components)
#print(W_init)
s_init=100
ppca_data, W, s_squared, S = ppca(data, num_components,W_init,s_init)


#eig_val=[]
#for v in W:
#    eigenvalue = np.dot(S, v) / v
#    eig_val.append(eigenvalue)
#indices = np.argsort(eig_val)[::-1]
#W = W[indices]
print(W)
#print(eig_vec.shape)
#print(pca_data.T.shape)
#print(ppca_data.shape)
#print(np.mean(np.asarray(data), axis=0))

#reconstructed_data_pca = np.dot(pca_data, eig_vec.T) + np.mean(np.asarray(data), axis=0)
reconstructed_data_ppca = np.dot(ppca_data, W.T) + np.mean(np.asarray(data),axis=0)

#print(reconstructed_data_pca[0])
print(reconstructed_data_ppca[0])
print(np.asarray(data)[0])'''

i=1
print(type(i) ==int)