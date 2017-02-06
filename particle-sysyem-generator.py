import numpy as np
from time import time


class RareEvents:
    def __init__(self, mu_0, score_function,
                 shaker,level, p_0 = 0.75):
        """
        :param level: level to estimate
        :param p_0: successful rate, the classical way is to choose p_0 large,
                    but for AMS, in order to fix the level, we would like to choose
                    p_0 small. This will also improve the quality of the variance 
                    estimator.
        :param mu_0: distribution of X_0
        :param score_function: black box score_function
        :param shaker: metro-polis/Gibbs/Gaussian(for the toy example) kernel
        """
        self.mu_0 = mu_0
        self.score_function = score_function
        self.shaker = shaker
        self.level = level
        self.p_0 = p_0

    def adaptive_levels(self,N, shake_times = 1, status_tracking = False):
        # Initiation
        t_0 = time()
        X = self.mu_0(N)
        xi = [X]
        A = [] 
        
        p_0 = self.p_0
        L = np.array([-np.Inf,np.sort(self.score_function(X))[np.int((1-p_0)*N)]])
        k = 1

        while(L[k]<self.level):
            I = []
            survive_index = []
            for i in range(N):
                if self.score_function(X[i])>L[k]:
                    survive_index += [i]

           # to ensure that I_k would not be empty
            if len(survive_index) == 0:        
                break

            A_k = np.zeros(N,dtype = np.int) 
            X_cloned = np.zeros(N)
            for i in range(N):
                A_k[i] = np.random.choice(survive_index)
                X_cloned[i] = X[A_k[i]]
            
            A += [A_k]   
            X = X_cloned

            
            for sigma_range in np.arange(0.35,0.05,-0.05):
                for j in range(N):
                    for index_shaker in range(shake_times):
                        X_iter = self.shaker(X[j],sigma_1 = sigma_range)
                    if self.score_function(X_iter)>L[k]:
                        X[j] = X_iter
            L = np.append(L, np.sort(self.score_function(X))[np.int((1-p_0)*N)])
            xi += [X]
            k += 1

        N_L = np.sum((self.score_function(X)>self.level))
        r_hat = N_L/float(N)
        p_hat = p_0**(k-1)*r_hat 

        if status_tracking ==True:
            print ("estimation of p: " + str(p_hat))
            print ('____________________________________________________________\n')
            print ("Time spent: %s s" %(time() - t_0) )
            #print ("score_function called: %s times" % S_called_times)
        output = {'p_hat':p_hat,  \
                  'A':A,\
                  'xi':xi,\
                 }    
        return output 



if __name__ == '__main__':

    from scipy.stats import norm
    from time import time
    def S_test(X):
        '''
        score function which is a black box
        '''
        return np.abs(X)
    
    def get_paramS_test(level_test = 8, p_0 = 0.75,status_tracking = True):
        '''
        This function returns the real values for the toy example 
        '''
        real_p = (1-norm.cdf(level_test))*2
        n_0 = int(np.floor(np.log(real_p)/np.log(p_0)))
        r = real_p/(p_0**n_0)
        sigma2_theoretical = n_0*(1-p_0)/p_0 + (1-r)/r
        #l = [-np.inf]
        #for k in range(1,n_0+1,1):
        #    l = np.append(l, norm.ppf(1 - p_0**k/2))
        #l_ideal = np.append(l, level_test)
    
        if status_tracking == True:
            print ("p_0 = " + str(p_0) + '\t n_0 =' + str(n_0) + "\t r = " + str(r))
            #print ("sequence of levels: "+ str(l_ideal))
            print ("real value of p: " + str(real_p))
            print ("relative variance(ideal): " + str(sigma2_theoretical))
        return real_p, sigma2_theoretical, n_0,r
    def mu_0_test(N):
        '''
        param n: the size of particles
        '''
        return np.random.normal(0,1,N)
    
    def shaker_gaussian(x,sigma_1=0.2):
        '''
        a reversible transition kernel for mu_0_test
        '''
        c = np.sqrt(1+sigma_1**2)
        return np.random.normal(x/c,sigma_1/c,1)
    
    #def shaker_metropolis(x,sigma_1):
    #    iter = np.random.uniform(x-sigma_1,x+sigma_1)
    #    if np.exp(-0.5*(iter**2-x**2))>np.random.rand(1):
    #   	return iter
    #    else:
    #    	return x
    
    
    print ('\n============================================================')
    # parameters 
    N_test = 100 
    p_0_test = 0.3 
    shaker = shaker_gaussian
    shake_times = 2 
    num_simulation = 200
    level_test = 4 
    test_info = '|num_particles_' + str(N_test) + '|' + \
            str(shaker).split(' ')[1] + '|shake_times_' + str(shake_times) 
            
    ################################################################################
    
    print ('Info: ' + test_info)
    params = get_paramS_test(level_test = level_test, p_0 = p_0_test,status_tracking = True)
    
    # definition of the RareEvents class
    rare_test = RareEvents(mu_0 = mu_0_test, score_function = S_test,\
    	level = level_test,shaker = shaker, p_0 = p_0_test)
    
    # test
    test_result = rare_test.adaptive_levels(N = N_test, shake_times = shake_times, status_tracking=True)
    # tracing the genealogical information
    A = test_result['A']
    xi = test_result['xi']


    # output .json for tree visulization 
    n,N = np.shape(xi)
    n -= 1
    
    # creat (parent, child) couples
    links = []
    for p in range(n):
        for i in range(N):
            for j in range(N):
                if A[p][j] == i:
                    links += [('X_'+str(p)+'^'+str(i),'X_'+str(p+1)+'^'+str(j))]
    # json 
    import json
    
    parents, children = zip(*links)
    root_nodes = {x for x in parents if x[2]==str(0)}
    for node in root_nodes:
        links.append(('Root', node))
    
    def get_nodes(node):
        d = {}
        d['name'] = node
        children = get_children(node)
        if children:
            d['children'] = [get_nodes(child) for child in children]
        return d
    
    def get_children(node):
        return [x[1] for x in links if x[0] == node]
    
    tree = get_nodes('Root')
    # print json.dumps(tree, indent=1)
    
    # output
    with open('data.json', 'w') as fp:
        json.dump(tree, fp, indent=1)
