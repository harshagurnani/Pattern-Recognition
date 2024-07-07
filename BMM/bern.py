import numpy as np
from sklearn.cluster import KMeans

class BernMM:
    '''Bernoulli mixture model'''
    def __init__(self, nz=3, nx=5, use_kmeans_init=False, **kwargs):
        if 'X_init' in kwargs:
            self.set_params(nz, nx, use_kmeans_init=use_kmeans_init, X=kwargs['X_init'])
        else:
            self.set_params(nz, nx, use_kmeans_init=use_kmeans_init, **kwargs)


    def set_params(self, nz, nx, use_kmeans_init, **kwargs ):
        '''
        set initial parameters
        '''
        self.nz = nz
        self.nx = nx
        
        if use_kmeans_init and 'X' in kwargs:
            X = kwargs['X']
            kmeans = KMeans(n_clusters=nz, random_state=0).fit(X)
            labels = kmeans.labels_

            # Initialize prior_z
            self.prior_z = np.bincount(labels) / len(labels)

            # Initialize mu_x
            self.mu_x = np.zeros((nz, nx))
            for k in range(nz):
                self.mu_x[k] = X[labels == k].mean(axis=0)
        
        else:
            self.prior_z = np.array(kwargs.get('prior_z', np.ones(nz) / nz)).reshape(-1)
            assert self.prior_z.shape[0] == self.nz
            assert np.isclose(np.sum(self.prior_z), 1)

            self.mu_x = np.array(kwargs.get('mu_x', np.random.rand(nz, nx))).reshape(nz, -1)
            assert self.mu_x.shape[1] == self.nx
            assert np.all(self.mu_x>=0) # means between 0 and 1
            assert np.all(self.mu_x<=1)

        self.sigma_x = np.array(kwargs.get('sigma_x', np.array([np.diag([self.mu_x[kk,ii]*(1-self.mu_x[kk,ii]) for ii in range(nx)]) for kk in range(nz)]))).reshape(nz, nx, -1)
        assert self.sigma_x.shape[2] == self.nx


    def sample(self, T=1 ):
        '''
        generate complete data - latents and observartions from BernMM model
        '''
        z = np.zeros(T)
        x = np.zeros((T,self.nx))
        z = np.random.choice( self.nz, size=T, p=self.prior_z)  # which class is the data being generated from 
        for kk in range(self.nz):
            use_t = np.where(z==kk)[0]
            x[use_t] = np.random.binomial([1 for _ in range(self.nx)], p=self.mu_x[kk], size=(len(use_t),self.nx)) 
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
                l0+= self.prior_z[kk]* np.prod( np.power( self.mu_x[kk], X[tt])*np.power( 1-self.mu_x[kk], 1-X[tt]) )
            ll += np.log(l0)
        return ll


    def e_step( self, X ):
        '''
        E-step: estimate posterior probabilities of z given observations and current parameters in GMM model
        '''
        T,_ = X.shape
        gamma = np.zeros((T,self.nz))
        
        for tt in range(T):
            for kk in range(self.nz):
                gamma[tt,kk] = self.prior_z[kk]* np.prod( np.power( self.mu_x[kk], X[tt])*np.power( 1-self.mu_x[kk], 1-X[tt]) )
            gamma[tt,:] = gamma[tt,:]/np.sum(gamma[tt,:])
        
        return gamma
    
    def m_step( self, X, gamma, update_prior=True, update_mu=True ):
        '''
        M-step: update parameters based on computed posterior probabilities of latent variable z
        '''
        T,_ = X.shape
        assert gamma.shape==(T,self.nz)

        N_k = np.sum(gamma, axis=0)
        if update_mu:
            self.mu_x = (gamma.T @ X) / N_k[:, np.newaxis]
        
            self.sigma_x = np.array([np.diag([self.mu_x[kk,ii]*(1-self.mu_x[kk,ii]) for ii in range(self.nx)]) for kk in range(self.nz)])

        if update_prior:
            self.prior_z = N_k/np.sum(N_k)
            #print(np.sum(self.prior_z))
            #assert np.sum(self.prior_z)==1

    def fit_params(self, X, n_steps=10, tol=1e-3, update_prior=True, update_mu=True, update_sigma=True, print_ll=10 ):
        '''
        Fit GMM parameters based on data likelihood (ELBO) maximization by iterative EM algorithm
        '''
        ll0=self.get_ll(X)
        if n_steps>0:
            ll = np.zeros(n_steps)
            for step in range(n_steps):        
                gamma  = self.e_step( X )
                self.m_step( X, gamma=gamma, update_prior=update_prior, update_mu=update_mu )
                ll1=self.get_ll(X)
                incr = ll1-ll0
                if incr < 0:
                    raise ValueError("Log-likelihood decreased during EM step")
                if incr<tol:
                    break
                if (step%print_ll)==0:
                    print(f'Step {step + 1}: Log-likelihood increase = {incr:.4f}')
                ll0 = ll1
                ll[step] = ll0
        else:
            incr=1000
            ll=[ll0]
            step=0
            while incr>tol:
                gamma  = self.e_step( X )
                self.m_step( X, gamma=gamma, update_prior=update_prior, update_mu=update_mu )
                ll1=self.get_ll(X)
                incr = ll1-ll0
                if incr < 0:
                    raise ValueError("Log-likelihood decreased during EM step")
                if (step%print_ll)==0:
                    print(f'Step {step + 1}: Log-likelihood increase = {incr:.4f}')
                ll0 = ll1
                ll.append(ll0)
                step+=1

        return ll



