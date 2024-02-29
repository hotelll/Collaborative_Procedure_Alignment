import numpy as np
from sklearn import metrics
import scipy.sparse as sp
import torch


def clust_rank(mat, distance='cosine'):
    s = mat.shape[0]
    loc = mat[:, -1]
    mat = mat[:, :-1]
    loc_dist = np.sqrt((loc[:, None] - loc[:, None].T)**2)

    orig_dist = metrics.pairwise.pairwise_distances(mat, mat, metric=distance)
    orig_dist = orig_dist * loc_dist
    np.fill_diagonal(orig_dist, 1e12)
    initial_rank = np.argmin(orig_dist, axis=1)
    
    # The Clustering Equation
    A = sp.csr_matrix((np.ones_like(initial_rank, dtype=np.float32), (np.arange(0, s), initial_rank)), shape=(s, s))
    A = A + sp.eye(s, dtype=np.float32, format='csr')
    A = A @ A.T

    A = A.tolil()
    A.setdiag(0)   
    return A


def get_clust(a):
    _, u = sp.csgraph.connected_components(csgraph=a, directed=True, connection='weak', return_labels=True)
    return u


def FINCH(data, distance='cosine'):
    n_frames = data.shape[0]
    time_index = (np.arange(n_frames) + 1.) / n_frames
    data = np.concatenate([data, time_index[..., np.newaxis]], axis=1)

    # Cast input data to float32
    data = data.astype(np.float32)
    
    adj = clust_rank(data, distance=distance)
    group = get_clust(adj)
    
    return group


if __name__ == '__main__':
    frame_num = 16
    feature_dim = 192
    
    fake_input = torch.randn([frame_num, feature_dim])
    c = FINCH(fake_input)
    
    print(1)