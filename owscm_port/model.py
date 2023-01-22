import torch
import pyro
import pyro.distributions as dist

def rbf_kernel_log(x_1, x_2, lengthscale):
    # CAREFUL! Reimplement difference!
    return torch.outer(x_1, x_2) / (lengthscale**2)

def expit(x):
    return torch.exp(x) / (1.0 + torch.exp(x))

def process_cov(log_cov, scale, noise = 0):
    return torch.exp(log_cov) * scale + noise * torch.eye(log_cov.shape[0])

def generate_lengthscale(shape, scale):
    lengthscale = pyro.sample("lengthscale", dist.InverseGamma(shape, scale))
    return lengthscale

def generate_scale(shape, scale):
    scale = pyro.sample("scale", dist.InverseGamma(shape, scale))
    return scale

def generate_noise(shape, scale):
    noise = pyro.sample("noise", dist.InverseGamma(shape, scale))
    return noise

def generate_binary_T(logit_t):
    t = pyro.sample("T", dist.Bernoulli(expit(logit_t)))
    return t

def generate_U(u_cov, n):
    u = pyro.sample("U", dist.MultivariateNormal(torch.zeros(n), u_cov))
    return u

def generate_X(var):
    x = pyro.sample("X", dist.Normal(0, var))
    return x

def generate_N(rate):
    # Poisson shifted by 1
    n = 1 + pyro.sample("N", dist.Poisson(rate))
    return n

def generate_params():
    theta_xt = pyro.sample("theta_xt", dist.Normal(0,1))
    theta_t0n = pyro.sample("theta_t0n", dist.InverseGamma(4,4))
    theta_t1n = pyro.sample("theta_t1n", dist.InverseGamma(4,4))
    theta_xn = pyro.sample("theta_xn", dist.InverseGamma(4,4))
    theta_t0w = pyro.sample("theta_t0w", dist.Normal(0,1))
    theta_t1w = pyro.sample("theta_t1w", dist.Normal(0,1))
    theta_xw = pyro.sample("theta_xw", dist.Normal(0,1))
    theta_nw = pyro.sample("theta_nw", dist.Normal(0,1))
    theta_t0y = pyro.sample("theta_t0y", dist.Normal(0,1))
    theta_t1y = pyro.sample("theta_t1y", dist.Normal(0,1))
    theta_xy = pyro.sample("theta_xy", dist.Normal(0,1))
    theta_ny = pyro.sample("theta_ny", dist.Normal(0,1))
    theta_wy = pyro.sample("theta_wy", dist.Normal(0,1))

    return theta_xt, theta_t0n, theta_t1n, theta_xn, theta_t0w, theta_t1w, theta_xw, \
            theta_nw, theta_t0y, theta_t1y, theta_xy, theta_ny, theta_wy

def generate_individual(x, t, n, theta_t0w, theta_t1w, theta_xw, theta_nw, 
                        theta_t0y, theta_t1y, theta_xy, theta_ny, theta_wy):

    w_mean = (1 - t) * theta_t0w + t * theta_t1w +  x * theta_xw + n * theta_nw
    w = pyro.sample("w", dist.Normal(w_mean, 1))
    y_mean = (1-t) * theta_t0y + t * theta_t1y + x * theta_xy + n * theta_ny + w * theta_wy
    y = pyro.sample("y", dist.Normal(y_mean, 1))
    return y

def generate_cluster(theta_xt, theta_t0n, theta_t1n, theta_xn, theta_t0w, theta_t1w, theta_xw, 
                    theta_nw, theta_t0y, theta_t1y, theta_xy, theta_ny, theta_wy):

    x = pyro.sample("x", dist.Normal(0,1))
    p_t = expit(theta_xt * x)
    t = pyro.sample("t", dist.Bernoulli(p_t))
    p_n = (1-t) * theta_t0n + t * theta_t1n + torch.abs(x * theta_xn)
    n = pyro.sample("n", dist.Poisson(p_n))

    if n > 0:
        with pyro.plate("ys", n):
            ys = generate_individual(x, t, n, theta_t0w, theta_t1w, theta_xw, theta_nw, 
                                    theta_t0y, theta_t1y, theta_xy, theta_ny, theta_wy)
        return ys
    else:
        return torch.Tensor([])

def generate_population(n_clusters):
    theta_xt, theta_t0n, theta_t1n, theta_xn, theta_t0w, theta_t1w, theta_xw, theta_nw, \
        theta_t0y, theta_t1y, theta_xy, theta_ny, theta_wy = generate_params()

    with pyro.plate("ys", n_clusters):
        ys = generate_cluster(theta_xt, theta_t0n, theta_t1n, theta_xn, theta_t0w, theta_t1w, 
                            theta_xw, theta_nw, theta_t0y, theta_t1y, theta_xy, theta_ny, theta_wy)

    return ys

def continuous_gp_ow(hyperparams, nX, nW, nC):
    pass


    