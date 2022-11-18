import torch
import gpytorch
import pyro

def causal_estimand(y, y_t):
    pass

def gaussian_process_effect_estimation(scm, scm_samples, intervention_assignment):
    m = len(scm_samples)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    for i in range(m):
        new_assignment = {}
        for var in scm:
            if scm_samples[i][var] == intervention_assignment[var]:
                new_assignment[var] = intervention_assignment[var]
            else:
                parents_assignment = []
                for node in scm[var]['parents']:
                    parents_assignment.append(new_assignment[node])
                parents_assignment = torch.Tensor(parents_assignment).reshape(1,-1)
                gp = scm[var]['model']
                new_assignment[var] = likelihood(gp(parents_assignment))[0,0]
        q_i = causal_estimand(new_assignment)
        
