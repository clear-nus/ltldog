import os, sys
import numpy as np
import torch

from .tl import parser

sys.setrecursionlimit(10000)
os.environ['TL_RECORD_TRACE'] = '1'

class DTL_Cont_Cons_Evaluator():
    def __init__(self, device="cuda"):
        self.device = device
        self.rules = {}
        # self.n_cons = 0
        self.prop_exp_list = []
        self.cons_func_list = []
        self.n_cons_func = 0
        self.formula_evaluator = None

    def _agg_cons_sub(self, con_groups:list) -> list:
        cons_flat = []
        assert type(con_groups) is list
        if len(con_groups) > 0:
                for c_group in con_groups:
                    assert type(c_group) is list
                    if len(c_group) < 1:
                        continue 
                    for c in c_group:
                        cons_flat.append(c)
        return cons_flat

    def set_atomic_props(self, con_groups: list):
        n_props = len(con_groups)
        assert n_props > 0
        self.cons_func_list = self._agg_cons_sub(con_groups)
        self.n_cons_func = len(self.cons_func_list)

        prop_exp_list = []
        n_prop_exp = 0

        rules = {}
        for i in range(n_props):
            c_group = con_groups[i]
            if len(c_group) < 1:
                continue

            rule = "("
            flg_hd = True
            for c in c_group:
                n_prop_exp +=1 
                if not flg_hd:
                    rule += " & "
                flg_hd = False
                prop_exp = "c"+str(n_prop_exp)
                prop_exp_list.append(prop_exp)
                rule += prop_exp
            rule += ")"
            rules["p"+str(i)] = rule

        self.rules = rules
        self.prop_exp_list = prop_exp_list
        self.ap_map = lambda x: prop_exp_list.index(x)

    def get_assignments(self, st: torch.tensor) -> torch.tensor:
        """
            calculate the assignments for a BATCH on every low level constraints

            Input
                st: torch.tensor (batch, time_steps, feature)

            Return
                assignment: torch.tensor (batch, num_constraints, time_steps)
        """
        assert self.n_cons_func > 0, "No low level constraints assigned yet."

        vals = [None]*self.n_cons_func
        for i in range(self.n_cons_func):
            val = self.cons_func_list[i](torch.permute(st, (1, 2, 0)))
            vals[i] = torch.transpose(val, 0, 1).unsqueeze(1)
        return torch.cat(vals, dim=1)

    def set_ltl_formula(self, ltl):
        self.ltl = ltl
        for key, val in self.rules.items():
            self.ltl = self.ltl.replace(key, val)
        self.formula_evaluator = parser.parse(self.ltl)

    def get_evaluations(self, assigns: torch.tensor, rho = 1e3) -> torch.tensor:
        return self.formula_evaluator(assigns, ap_map=self.ap_map, rho=rho)
