from typing import *
import math
import gpytorch
import pyro
import torch
from pyro.infer.mcmc import NUTS, MCMC
from matplotlib import pyplot as plt
from relational import *

class NodeGPModel(gpytorch.models.ExactGP):
    
    def __init__(self, train_x, train_y, likelihood, parents):
        super().__init__(train_x, train_y, likelihood)
        self.parents = parents
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = self.compose_kernels()

    def compose_kernels(self):
        parents_kernels = []
        for p in self.parents:

            # Self edges are between attributes in the same entity type
            # One to many edges use the same kernel, just with the parent entity's attributes as arguments instead
            if self.parents[p] in ['self', 'one_to_many']:
                parents_kernels.append(gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()))

            # For many to one relationships, each instance could have a different no. of parents in ground graph
            # Many to many edges use the same kernel, just with the parent entity's attributes as arguments instead
            elif self.parents[p] in ['many_to_one', 'many_to_many']:
                parents_kernels.append(MultiSetKernel(self.parents))

        full_kernel = gpytorch.kernels.ProductStructureKernel(parents_kernels)
        return full_kernel                

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class MultiSetKernel(gpytorch.kernels.Kernel):

    def __init__(self, s, lamb, adj_mat_dict):
        '''
        Check gpytorch docs for how to implement init and forward
        '''
        super().__init__() 
        self.s = s
        self.lamb = lamb 
        self.adj_mat_dict = adj_mat_dict 

    def _distance(self, node1, node2, relation, data):
        distance = 0
        adj_mat = self.adj_mat_dict[relation]
        # Get instances that are related through each node through the given relationship class
        node1_instances = list(adj_mat.loc[node1.instance, adj_mat.loc[node1.instance] == True].index)
        node2_instances = list(adj_mat.loc[node2.instance, adj_mat.loc[node2.instance] == True].index)
        for n1 in node1_instances:
            for n2 in node2_instances:
                value1 = data[node1.entity][node1.attribute][node1.instance]
                value2 = data[node2.entity][node2.attribute][node2.instance]
                distance += (value1 - value2) ** 2
        distance /= (len(node1_instances) * len(node2_instances))
        return distance

    def distance(self, node1, node2, relation, adj_mat_dict, data):
        return self._distance(node1, node2, relation, adj_mat_dict, data) \
                - self._distance(node1, node1, relation, adj_mat_dict, data)/2 \
                - self._distance(node2, node2, relation, adj_mat_dict, data)/2

    def forward(self, node1, node2, relation, adj_mat_dict, data, **params):
        return self.s * torch.exp(-self.distance(node1, node2, relation, adj_mat_dict, data)/self.lamb)


    
