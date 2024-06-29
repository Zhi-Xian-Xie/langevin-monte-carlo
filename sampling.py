import numpy as np
import torch
import matplotlib.pyplot as plt
#from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm

def unadjusted_langevin_algorithm_gpu(potential, n_samples=100000, step=0.01, iters=10000):
    if not torch.cuda.is_available():
        print("No GPU")
        return
    
    device=torch.device('cuda')
    Z0 = torch.randn(n_samples, 2, device=device)
    Zi = Z0
    for i in tqdm(range(iters)):
        Zi.requires_grad_()
        u = potential(Zi).sum()
        grad = torch.autograd.grad(u, Zi)[0]
        #print(grad)
        Zi = Zi.detach() - step * grad + np.sqrt(2 * step) * torch.randn(n_samples, 2, device=device)
        #samples.append(Zi.detach().cpu().numpy())
    return Zi.detach().cpu().numpy()

def log_Q_gpu(potential, z_prime, z, step):
    z.requires_grad_()
    grad = torch.autograd.grad(potential(z).sum(), z)[0]
    return -(torch.norm(z_prime - z + step * grad, p=2, dim=1) ** 2) / (4 * step)

def metropolis_adjusted_langevin_algorithm_gpu(potential, n_samples=100000, step=0.1, iters=10000):
    if not torch.cuda.is_available():
        print("No GPU")
        return
    
    torch.cuda.empty_cache()
    device=torch.device('cuda')
    Z0 = torch.randn(n_samples, 2, device=device)
    Zi = Z0
    for i in tqdm(range(iters)):
        #torch.cuda.empty_cache()
        Zi.requires_grad_()
        u = potential(Zi).sum()
        grad = torch.autograd.grad(u, Zi)[0]
        #print(grad)
        prop_Zi = Zi.detach() - step * grad + np.sqrt(2 * step) * torch.randn(n_samples, 2, device=device)
        log_ratio = -potential(prop_Zi) + potential(Zi) +\
                    log_Q_gpu(potential, Zi, prop_Zi, step) - log_Q_gpu(potential, prop_Zi, Zi, step)
        mask= torch.rand(n_samples,device=device) < torch.exp(log_ratio)
        #print(mask)
        mask=mask.double().view(n_samples,-1)
        Zi=mask*prop_Zi + (1-mask)*Zi.detach()
        #samples.append(Zi.detach().cpu().numpy())
    return Zi.detach().cpu().numpy()