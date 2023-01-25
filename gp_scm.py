from typing import *
import torch
import gpytorch

# Import classes and functions for relational models
from relational import *

class RelationalGPModel:

    def __init__(self, scm: RelationalSCM, skeleton: RelationalSkeleton):
        # Store kernels for each attribute in every entity in the model
        # Key is [entity_name][attribute_name]
        self.kernels_dict = {}
        # Store adjacency matrices for each relation between entities in the model
        # Key is [relation_name]
        self.adj_mat_dict = create_adj_mat_dict(scm.structure, skeleton) 
        # Store hyperparams for all functions in the model
        # 
        self.hyperparams = {}
        self.scm = scm

    def get_attribute_vector(self, entity, attribute, skeleton: RelationalSkeleton):
        attribute_instances = skeleton.entity_instances[entity][attribute]
        return torch.Tensor(attribute_instances)

    def get_cov_noiseless(self, entity_name, attribute_name):
        '''
        Compute the overall kernel
        '''
        incoming_edges = self.scm.structure.get_incoming_edges(entity_name, attribute_name)
        
        # Replace these!
        n = 10
        cov_noiseless = np.zeros((n,n))

        kernels_dict = {}
        parents_kernels = []

        for relation, edge in incoming_edges:
            if not relation in kernels_dict.keys():
                kernels_dict[relation] = {}

        lengthscale = self.hyperparams[entity_name][attribute_name]["lengthscale"]
        outputscale = self.hyperparams[entity_name][attribute_name]["outputscale"]


        # For self edges, just obtain the full covariance matrix
        if relation == "self":
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            kernel.lengthscale = lengthscale
            kernel.outputscale = outputscale
            parents_kernels.append(kernel)

        
