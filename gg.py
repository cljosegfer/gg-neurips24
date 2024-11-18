
import torch
import numpy as np

class GG:
    def __init__(self, device: str) -> None:
        self.device = device

    def torch(self, X: np.ndarray):
        X = torch.Tensor(X).to(self.device)
        n = X.shape[0]
        F = torch.cdist(X, X, p = 2)**2
        F.fill_diagonal_(float('inf'))

        adj = torch.zeros((n,n), dtype=torch.bool).to(self.device)
        for i in (range(n-1)):
            A = F[i]+F[i+1:]
            idx_min = torch.argmin(A, axis=1)
            a = A[torch.arange(A.shape[0]), idx_min] - F[i, i+1:]
            adj[i, i+1:] = torch.where(a > 0, 1, 0)
        adj = adj + adj.T
        return adj
    
    def bootstrap(self, X: np.ndarray, btsz: float, epochs: int):
        N = X.shape[0]
        idx = np.arange(N)
        adjb = torch.ones((N, N), dtype=torch.bool)

        for epoch in (range(epochs)):
            np.random.shuffle(idx)
            for b in range(0, N, btsz):
                idx_batch = idx[b:min(b+btsz, N)]
                X_batch = X[idx_batch, :]
                adjb[np.ix_(idx_batch, idx_batch)] *= self.torch(X_batch)
        return adjb