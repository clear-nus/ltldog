from typing import Union, List
import logging
import torch
import torch.nn as nn
import numpy as np
import einops
from einops.layers.torch import Rearrange

from diffusion_policy.model.diffusion.conv1d_components import (
    Downsample1d, Conv1dBlock)
from diffusion_policy.model.diffusion.conditional_unet1d import (
    ConditionalResidualBlock1D)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.diffusion.conditional_unet1d import (
    ConditionalUnet1D)

import dgl
import pdb
from diffusion_policy.common.ast_builder import ASTBuilder
from diffusion_policy.common.ltl_parser import LTLParser, LTLParseError
from diffusion_policy.model.gnns.graphs.RGCN import RGCNRootShared
from diffusion_policy.model.gnns.graphs.GNN import GNNMaker

logger = logging.getLogger(__name__)


class ConditionalValueUnet1D(nn.Module):
    def __init__(self,
        horizon,
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
        ltl_embed_input_dim=22,
        ltl_embed_output_dim=32,
        ltl_embed_hidden_dim=32,
        ltl_embed_num_layers=8,
        propositions: List[str]=None,
        output = print,
        legacy=True,
        ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                # down encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                # up encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale)
            ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))
        
        # final_conv = nn.Sequential(
        #     Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
        #     nn.Conv1d(start_dim, input_dim, 1),
        # )
            
        # fc_dim = down_dims[-1] * max(horizon, 1)
        fc_dim = down_dims[-1]

        self.ltl_embed_output_dim = ltl_embed_output_dim
        self.gnn = RGCNRootShared(
            ltl_embed_input_dim, 
            ltl_embed_output_dim,
            hidden_dim=ltl_embed_hidden_dim, 
            num_layers=ltl_embed_num_layers
        )

        if not legacy: 
            propositions = list(set(list(propositions+['0', '1', 'True', 'False'])))
            self.str2tup_converter = LTLParser(propositions=propositions)
            self.graph_builder = ASTBuilder(propositions=propositions)

        self.final_block = nn.Sequential(
            nn.Linear(fc_dim + ltl_embed_output_dim + cond_dim, fc_dim // 2),
            nn.Mish(),
            nn.Linear(fc_dim // 2, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.down_modules = down_modules

        self.output = output
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int],
            formulas:Union[List[str], List[dgl.DGLGraph]], legacy=True,
            local_cond=None, global_cond=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        sample = einops.rearrange(sample, 'b h t -> b t h')

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)
        
        # encode local features
        h_local = list()
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)
        
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        '''
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)
        '''

        ## TODO: check whether this works
        ##      if not, try using nn.AvgPool1d for lowering the dimension 
        x = torch.mean(x, dim=-1)
        trj_embed = x.view(len(x), -1)

        if (not legacy):
            ## Conversion upon inference
            formula_tups = [self.str2tup_converter(form_str) for form_str in formulas]
            ## Peformance tip: preprocess4gnn is very slow
            formula_graphs = self.preprocess4gnn(formula_tups, legacy=legacy)
        else:
            formula_graphs = formulas
        formula_embed = self.gnn(formula_graphs)

        embeds = torch.cat([trj_embed, formula_embed, global_feature], dim=-1)

        out = self.final_block(embeds)
        return out

    def preprocess4gnn(self, inputs, ast=None, device=None, 
                       legacy=False) -> np.ndarray:
        """
        This function receives the LTL formulas and convert them into inputs 
        for a GNN. 

        Args:
            texts (`List[str]`): 
                LTL formulas to be converted.  
            ast:
                A converter for converting a single formula to a graph; 
                Default: `self.graph_builder`.  
            device: 
                'cuda' or 'cpu'; 
                Default: `self.device`.
        """
        ast = self.graph_builder if ast is None else ast
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        ## Performance bug fixed, no need to preprocess and load then load the graphs. 
        ## Set legacy to True to use the old version.
        if not legacy:
            graphs = np.array([ast(tup, dim_is_root=self.ltl_embed_output_dim).to(device) for tup in inputs])
        else:
            logger.info("**UserWarning: Using legacy method for processing graphs, performance is poor.")
            graphs = np.array([[ast(tup).to(device)] for tup in inputs])
            for i in range(graphs.shape[0]):
                d = graphs[i][0].nodes[None].data['feat'].size()
                # pdb.set_trace()
                root_weight = torch.ones((1, self.ltl_embed_output_dim))
                others_weight = torch.zeros((d[0]-1, self.ltl_embed_output_dim))
                weight = torch.cat([root_weight, others_weight])
                graphs[i][0].nodes[None].data['is_root'] = weight.to(self.device)

        return graphs

