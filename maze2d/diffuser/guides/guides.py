import torch
import torch.nn as nn
import pdb

import diffuser.models as models


class ValueGuide(nn.Module):

    def __init__(self, model, graph=None):
        super().__init__()
        self.model = model
        self.graph = graph

    def set_graph(self, graph):
        self.graph = graph

    def forward(self, x, cond, t):
        if self.graph is None:
            output = self.model(x, cond, t)
        else:
            output = self.model(x, cond, t, self.graph)
        return output.squeeze(dim=-1)

    def gradients(self, x, *args):
        x.requires_grad_()
        y = self(x, *args)
        grad = torch.autograd.grad([y.sum()], [x])[0]
        x.detach()
        return y, grad


class PSGuide(nn.Module):

    def __init__(self, evaluator):
        super().__init__()
        self.evaluator = evaluator

    def set_ltl_formula(self, ltl):
        self.evaluator.set_ltl_formula(ltl)

    def forward(self, x_0):
        return self.evaluator.get_evaluations(self.evaluator.get_assignments(x_0))

    def gradients(self, x_prev, x_0_hat, *args):
        y = self(x_0_hat)
        grad = torch.autograd.grad([y.sum()], [x_prev])[0]
        return y, grad
