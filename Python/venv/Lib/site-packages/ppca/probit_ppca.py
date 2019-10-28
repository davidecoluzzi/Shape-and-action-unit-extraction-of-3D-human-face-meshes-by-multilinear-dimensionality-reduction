import numpy as np
from scipy.stats import norm

## Simon's book Chapter 7.7
## apply probit function to solve PPCA for binary data
class probitPPCA:
    ## update W, X, and Q when doing inference
    ## Y: input binary data with shape (N, M), values should be 1 and -1. Use 0 to represent missing data
    ## D: number of ppca components
    ## Q: real-value variable to do variational approximation
    ## Z: represent missing data or not in Y, z == 1 means value not missing, z == 0 means missing
    def __init__(self, D = 2, n_iters = 100, verbose = False):
        self.D = D
        self.n_iters = n_iters
        self.verbose = verbose
        
    def _init_paras(self, N, M, D):
        self.a = np.ones((N, M)) * -1e6
        self.b = np.ones((N, M)) * 1e6

        self.e_X = np.zeros((N, D))
        self.e_w = np.random.randn(M, D)

        self.e_XXt = np.zeros((D,D,N))
        self.e_wwt = np.zeros((D,D,M))

        for n in range(N):
            self.e_XXt[:, :, n] = self.e_X[n, :].reshape(D, 1).dot(self.e_X[n, :].reshape(1, D))
            
        self.Q = np.zeros((N,M))
        self.bias = np.zeros((N,M))
    
    def _update_Q(self):
        xw = self.e_X.dot(self.e_w.T) + self.bias
        ai = self.a - xw
        bi = self.b - xw
        self.Q = xw + (norm.pdf(ai) - norm.pdf(bi)) / (norm.cdf(bi) - norm.cdf(ai))

    def _update_W(self, M, D):
        for m in range(M):
            covw_m = np.linalg.inv(np.eye(D) + np.sum(self.e_XXt * np.tile(self.Z[:, m], (D, D, 1)), axis=2))

            ## forgot to multiply Z, fixed now!
            self.e_w[m, :] = covw_m.dot(np.sum(self.e_X * np.matlib.repmat(self.Z[:, m].reshape(-1, 1) * self.Q[:, m].reshape(-1, 1), 1,D), axis = 0).reshape(D, 1)).reshape(D)
            self.e_wwt[:, :, m] = covw_m + self.e_w[m,:].reshape(D,1).dot(self.e_w[m,:].reshape(1,D))

    def _update_X(self, N, D):
        for n in range(N):
            covx_n = np.linalg.inv(np.eye(D) + np.sum(self.e_wwt * np.tile(self.Z[n, :], (D, D, 1)), axis = 2))
            self.e_X[n, :] = covx_n.dot(np.sum(self.e_w.T * np.matlib.repmat(self.Z[n, :] * self.Q[n,:], D, 1), axis = 1).reshape(D, 1)).reshape(D)
            self.e_XXt[:, :, n] = covx_n + self.e_X[n,:].reshape(D,1).dot(self.e_X[n,:].reshape(1,D))

    def _update_bias(self):
        self.bias = 0.5 * (self.Q - self.e_X.dot(self.e_w.T))

    def _update(self, Y, N, M, D):
        self._update_Q()
        self._update_bias()
        self._update_X(N, D)
        self._update_W(M, D)
        
    def fit(self, Y):
        N, M = Y.shape
        D = self.D
        self._init_paras(N, M, D)
        self.a[Y==1] = 0
        self.b[Y==-1] = 0
        self.Z = (Y != 0).astype(float)
        
#         temporarily comment these two lines out
#         if not D:
#             D = N

        for it in range(self.n_iters):
            self._update(Y, N, M, D)

    def recover(self):
        return norm.cdf(self.e_X.dot(self.e_w.T) + self.bias)
