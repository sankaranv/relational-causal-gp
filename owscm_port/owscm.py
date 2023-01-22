import torch
import pyro

def squared_difference(x_1, x_2):
    return (x_1.reshape(1, -1) -  x_1.reshape(-1, 1)) ** 2

def unnormalized_rbf(x_1, x_2):
    return torch.exp(-squared_difference(x_1, x_2))

def rbf(x_1, x_2, lengthscale, scale):
    return torch.exp(-squared_difference(x_1, x_2) / lengthscale) * scale

def log_rbf(x_1, x_2, lengthscale, scale):
    return torch.log(torch.Tensor([scale])) - squared_difference(x_1, x_2) / lengthscale

def set_kernel(x_1, x_2):
    kernel = torch.zeros((x_1.shape[0], x_1.shape[0]))
    for i in range(x_1.shape[1]):
        kernel += squared_difference(x_1[:,1], x_2[:,1])
    return kernel / x_1.shape[1]

def normalized_set_rbf(x_1, x_2, lengthscale, scale):
    return scale * torch.exp((set_kernel(x_1, x_2) - 0.5 * (set_kernel(x_1, x_1) + set_kernel(x_2, x_2))) / lengthscale)

def create_unnormalized_cov_dict(relational_schema, relational_skeleton):
    unnormalized_cov_dict = {}
    for entity in relational_schema.entities:
        attribute_dict = {}
        entity_instances = relational_skeleton

    return unnormalized_cov_dict