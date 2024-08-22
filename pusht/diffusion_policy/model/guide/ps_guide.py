import torch
import torch.nn as nn
import pdb

from diffusion_policy.dtl.dtl_cont_cons import DTL_Cont_Cons_Evaluator


class PSGuide(nn.Module):
    """
    Posterior sampling guide for the Diffusion Policy.
    
    The guide uses the value and gradients of the DTL evaluator for guidiing. 
    """
    def __init__(self, evaluator: DTL_Cont_Cons_Evaluator):
        super().__init__()
        self.evaluator = evaluator

    def set_ltl_formula(self, ltl):
        self.evaluator.set_ltl_formula(ltl)

    def forward(self, x_0, *args, **kwargs):
        return self.evaluator.get_evaluations(
            self.evaluator.get_assignments(x_0, *args, **kwargs))

    def gradients(self, x_prev, x_0_hat, *args, **kwargs):
        normalized = kwargs.get("normalized", False)
        custom_normalizer = kwargs.get("custom_normalizer", None)
        y = self(x_0_hat, normalized=normalized, 
                 custom_normalizer=custom_normalizer)
        grad = torch.autograd.grad([y.sum()], [x_prev])[0]
        return y, grad
