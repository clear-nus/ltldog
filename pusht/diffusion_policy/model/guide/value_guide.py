import torch
import torch.nn as nn
import pdb

from diffusion_policy.model.diffusion.value_classifier import ConditionalValueUnet1D


class ValueGuide(nn.Module):

    def __init__(self, 
                 model, 
                 ltl=None,
                 graph:list=None):
        super().__init__()
        self.model = model
        self.ltls = ltl
        self.graph = graph

    def set_graph(self, graph):
        self.graph = graph

    def forward(self, x, *args, **kwargs):
        if self.graph is None:
            output = self.model(x, *args, **kwargs)
        else:
            if x.shape[0] != len(self.graph):
                self.graph = [self.graph[0]] * x.shape[0]
            output = self.model(x, *args, formulas=self.graph, **kwargs)
        return output.squeeze(dim=-1)

    def gradients(self, x, *args, **kwargs):
        if isinstance(self.model, ConditionalValueUnet1D):
            return self.gradients_batch(x, *args, **kwargs)
        else:
            return self.gradients_loop(x, *args, **kwargs)

    def gradients_batch(self, x, *args, **kwargs):
        x.requires_grad_()
        y = self(x, *args, **kwargs)
        grad = torch.autograd.grad([y.sum()], [x])[0]
        x.detach()
        return y, grad

    def gradients_loop(self, x, *args):
        '''
            interface for direct use of formula function, not working
        '''
        bs = x.shape[0]
        ys = []
        grads = []
        for i in range(bs):
            xi = x[i].clone().detach()
            xi.requires_grad = True
            y = self.model(xi)
            ys.append(y)
            grads.append(torch.autograd.grad([y], [xi])[0])
        return torch.stack(ys), torch.stack(grads)

