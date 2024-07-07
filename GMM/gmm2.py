import numpy as np
import scipy.linalg as lin

class GMM:
    '''
    Optimized with vector computations
    '''
    def __init__(self, nz=2, nx=10, **kwargs):
        self.set_params( nz, nx, **kwargs )

    def set_params(self, nz, nx, **kwargs ):
        '''
        set initial parameters
        '''
        self.nz = nz
        self.nx = nx
        
        self.prior_z = np.array(kwargs.get('prior_z', np.ones(nz) / nz)).reshape(-1)
        assert self.prior_z.shape[0] == self.nz
        assert np.isclose(np.sum(self.prior_z), 1)

        self.mu_x = np.array(kwargs.get('mu_x', np.zeros((nz, nx)))).reshape(nz, -1)
        assert self.mu_x.shape[1] == self.nx

        self.sigma_x = np.array(kwargs.get('sigma_x', np.array([np.eye(nx) * np.random.rand() for _ in range(nz)]))).reshape(nz, nx, -1)
        assert self.sigma_x.shape[2] == self.nx

        self.precision = np.linalg.inv(self.sigma_x)


    def sample(self, T=1 ):
        '''
        generate complete data - latents and observartions from GMM model
        '''
        z = np.zeros(T)
        x = np.zeros((T,self.nx))
        z = np.random.choice( self.nz, size=T, p=self.prior_z)
        for kk in range(self.nz):
            use_t = np.where(z==kk)[0]
            x[use_t] = np.random.multivariate_normal( mean=self.mu_x[kk], cov=self.sigma_x[kk], size=len(use_t))
        return z, x
    
    def get_ll( self, X ):
        '''
        get data likelihood under current params
        '''
        T, _ = X.shape
        ll = 0
        for tt in range(T):
            l0 = 0
            for kk in range(self.nz):
                l0+= self.prior_z[kk]* 1/(2*np.pi)**(self.nx/2) * 1/lin.det(self.sigma_x[kk])**(1/2) \
                                * np.exp(-1/2 * (X[tt,:]-self.mu_x[kk])@self.precision[kk]@(X[tt,:]-self.mu_x[kk]).T )
            ll += np.log(l0)
        return ll


    def e_step( self, X ):
        '''
        E-step: estimate posterior probabilities of z given observations and current parameters in GMM model
        '''
        T, _ = X.shape
        gamma = np.zeros((T, self.nz))
        
        for kk in range(self.nz):
            diff = X - self.mu_x[kk]
            exponent = -0.5 * np.sum(diff @ self.precision[kk] * diff, axis=1)
            coef = self.prior_z[kk] / np.sqrt((2 * np.pi) ** self.nx * np.linalg.det(self.sigma_x[kk]))
            gamma[:, kk] = coef * np.exp(exponent)
        
        gamma /= gamma.sum(axis=1, keepdims=True)
        return gamma
    
    def m_step( self, X, gamma, update_prior=True, update_mu=True, update_sigma=True ):
        '''
        M-step: update parameters based on computed posterior probabilities of latent variable z
        '''
        T, _ = X.shape
        N_k = gamma.sum(axis=0)
        
        if update_mu:
            self.mu_x = (gamma.T @ X) / N_k[:, np.newaxis]
        
        if update_sigma:
            for kk in range(self.nz):
                diff = X - self.mu_x[kk]
                self.sigma_x[kk] = (gamma[:, kk][:, np.newaxis, np.newaxis] * (diff[:, :, np.newaxis] @ diff[:, np.newaxis, :])).sum(axis=0) / N_k[kk]
                self.precision[kk] = np.linalg.inv(self.sigma_x[kk])

        if update_prior:
            self.prior_z = N_k/np.sum(N_k)
            #print(np.sum(self.prior_z))
            #assert np.sum(self.prior_z)==1

    def fit_params(self, X, n_steps=10, tol=1e-3, update_prior=True, update_mu=True, update_sigma=True ):
        '''
        Fit GMM parameters based on data likelihood (ELBO) maximization by iterative EM algorithm
        '''
        ll0=self.get_ll(X)
        if n_steps>0:
            ll = np.zeros(n_steps)
            for step in range(n_steps):        
                gamma  = self.e_step( X )
                self.m_step( X, gamma=gamma, update_prior=update_prior, update_mu=update_mu, update_sigma=update_sigma )
                ll1=self.get_ll(X)
                incr = ll1-ll0
                assert incr>0
                print('increase = '+np.str_(incr))
                ll0 = ll1
                ll[step] = ll0
        else:
            incr=1000
            ll=[ll0]
            while incr>tol:
                gamma  = self.e_step( X )
                self.m_step( X, gamma=gamma, update_prior=update_prior, update_mu=update_mu, update_sigma=update_sigma )
                ll1=self.get_ll(X)
                incr = ll1-ll0
                assert incr>=0
                #print('increase = '+np.str_(incr))
                ll0 = ll1
                ll.append(ll0)

        return ll



