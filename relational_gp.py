import numpy as np
import GPy

class RelationalGP:

    def __init__(self, lengthscale = 0.1, variance = 0.1):
        self.lengthscale = lengthscale
        self.variance = variance
        self.params = {}

    def build_param_dict(self):
        pass

    def model(self):
        state_policy_kernel = GPy.kern.RBF(input_dim = 1, 
                                           variance = self.params['state']['policy']['variance'],
                                           lengthscale = self.params['state']['policy']['lengthscale'])
        town_policy_kernel = GPy.kern.RBF(input_dim = 1, 
                                          variance = self.params['town']['policy']['variance'],
                                          lengthscale = self.params['town']['policy']['lengthscale'])
        town_prevalence_kernel = GPy.kern.RBF(input_dim = 1, 
                                              variance = self.params['town']['prevalence']['variance'],
                                              lengthscale = self.params['town']['prevalence']['lengthscale'])  
        business_occupancy_kernel = GPy.kern.RBF(input_dim = 1, 
                                                 variance = self.params['business']['policy']['variance'],
                                                 lengthscale = self.params['business']['policy']['lengthscale'])

                            
                                            
