import torch

import model

def conditional_ITE(uy_lengthscale, ty_lengthscale, xy_lengthscale, y_noise, y_scale, u, x, t, y, do_t):
    uy_cov_log = torch.sum(model.rbf_kernel_log(u, u, uy_lengthscale))
    xy_cov_log = torch.sum(model.rbf_kernel_log(x, x, xy_lengthscale))
    ty_cov_log = torch.sum(model.rbf_kernel_log(t, t, ty_lengthscale))
    