import torch
import gpytorch
import pyro
import typing
from relational import *

def causal_estimand(y, y_t):
    pass

def gaussian_process_effect_estimation(scm: RelationalSCM, num_samples: int, input_data: dict, intervention: dict) -> float:
    
    """Obtain samples from the posterior over causal effects

    Args:
        scm (RelationalCausalModel): parameters for the relational causal model
        num_samples (int): number of samples used for Monte Carlo approximation
        intervention_assignment (dict): intervention assignment for variables in the SCM

    Returns:
        float: Monte Carlo estimate of the causal effect
    """

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    causal_effect = 0
    for i in range(num_samples):
        # Initialize dict for storing counterfactual outcomes
        intervention_assignment = {}
        for var in scm.topological_ordering:
            if var in intervention:
                # If the variable has been intervened on, then set the value of the attribute to the intervened value
                num_instances = input_data[var].shape[0]
                intervention_assignment[var] = torch.full(size = (num_instances,), fill_value = intervention[var])
            else:
                # If the variable has not been intervened on, sample a value for it from the posterior
                # First get the values for the parents, this will be the input to the model
                parents_assignment = []
                for node in scm[var]['parents']:
                    parents_assignment.append(intervention_assignment[node])
                parents_assignment = torch.Tensor(parents_assignment).reshape(1,-1)
                model = scm[var]['model']
                intervention_assignment['var'] = likelihood(model(parents_assignment))[0,0]
        # Compute a sample for the causal estimand using the input data and the counterfactual outcome
        causal_effect += causal_estimand(input_data, intervention_assignment)
    
    # Return MC estimate for causal effect
    causal_effect /= num_samples
    return causal_effect
