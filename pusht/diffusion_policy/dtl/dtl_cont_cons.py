import numpy as np
import torch
import sys, os

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .tl import parser

# Raise the recursion limit to avoid problems when parsing formulas
sys.setrecursionlimit(10000)
# Setting TL_RECORD_TRACE asks DTL to record the evaluation trace.
# Using this we can find the conflicting part between logits and formula.
os.environ['TL_RECORD_TRACE'] = '1'


class DTL_Cont_Cons_Evaluator():
    def __init__(self, device="cuda"):
        self.device = device
        self.rules = {}
        self.rule_str = ""
        # self.n_cons = 0
        self.prop_exp_list = []
        self.cons_func_list = []
        self.n_cons_func = 0
        self.cons_val = None
        self.formula_evaluator = None

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

    def get_assignments(self, st: torch.tensor, 
                        normalized:bool = False, 
                        custom_normalizer = None) -> torch.tensor:
        """
            calculate the assignments for a BATCH on every low level constraints

            Input
                st: torch.tensor (batch, time_steps, feature)
                normalized: bool (default: False)
                custom_normalizer: a normalizer object (default: None)

            Return
                assignment: torch.tensor (batch, num_constraints, time_steps)
        """
        assert self.n_cons_func > 0, "No low level constraints assigned yet."

        vals = [None]*self.n_cons_func
        for i in range(self.n_cons_func):
            val = self.cons_func_list[i](
                torch.permute(st, (1, 2, 0)), 
                normalized=normalized,
                custom_normalizer=custom_normalizer
            )
            vals[i] = torch.transpose(val, 0, 1).unsqueeze(1)
        return torch.cat(vals, dim=1)

    def set_ltl_formula(self, ltl):
        self.ltl = ltl
        for key, val in self.rules.items():
            self.ltl = self.ltl.replace(key, val)
        self.formula_evaluator = parser.parse(self.ltl)

    def get_evaluations(self, assigns: torch.tensor, rho = 1e3) -> torch.tensor:
        return self.formula_evaluator(assigns, ap_map=self.ap_map, rho=rho)


    def generate_formula(self, neg_cons, pos_cons) -> str:
        self.rule_str = ""
        self.prop_exp_list = []
        cnt_prop = 0

        # Generate negative constraints (something that should never happen)
        neg_rule = ""
        neg_prop_exp_list = []
        neg_rule, neg_prop_exp_list, cnt_prop = \
            self._gen_subform(neg_cons, negate=True, cnt_prop=cnt_prop)
        
        # Generate positive constraints (something that should always happen)
        pos_rule = ""
        pos_prop_exp_list = []
        pos_rule, pos_prop_exp_list, cnt_prop = \
            self._gen_subform(pos_cons, negate=False, cnt_prop=cnt_prop)
        
        # Merge both together
        # self.n_cons = cnt_prop
        self.prop_exp_list = neg_prop_exp_list + pos_prop_exp_list
        self.rule_str = "(" \
            + (neg_rule if len(neg_rule)>0 else "") \
            + ((" & " + pos_rule) if len(pos_rule)>0 else "") \
            + ")"

        # Construct the evaluator for the rule
        self.formula_evaluator = parser.parse(self.rule_str)
        self.aggregate_constraint(neg_cons, pos_cons)

        return self.rule_str


    def aggregate_constraint(self, neg_cons, pos_cons) -> list:
        cons_flat = []
        neg_cons_flat = self._agg_cons_sub(neg_cons)
        pos_cons_flat = self._agg_cons_sub(pos_cons)
        cons_flat += neg_cons_flat + pos_cons_flat
        
        self.cons_func_list = cons_flat
        return self.cons_func_list


    def evaluate_constraints(self, st, cons_funcs: list = []) -> torch.tensor:
        if len(cons_funcs) < 1:
            cons_funcs = self.cons_func_list
        self.cons_val = None

        for fn in cons_funcs:
            # assuming every `fn` outputs a 1xT tensor, 
            # where T is the trajectory length.
            # val = fn(st).unsqueeze(0)    # promote dim
            val = torch.vmap(fn)(st).unsqueeze(1)    # vectorize fn for batched inputs
            # print(f"val= {val}")
            self.cons_val = val if self.cons_val is None else\
                torch.cat((self.cons_val, val), dim=1)

        return self.cons_val


    def formula_evaluation(self, st:torch.tensor, rho = 1e3) -> torch.tensor:
        self.evaluate_constraints(st)
        return self._form_eval(rho=rho)


    def _form_eval(self, 
            rule: str = "", 
            prop_vals: torch.tensor = None, 
            list_of_props: list = None, 
            rho: float = 1e3
    ) -> torch.tensor:
        if len(rule) < 1:
            rule = self.rule_str
            evaluator = self.formula_evaluator
        else:
            evaluator = parser.parse(rule)
        assert evaluator is not None, "No rule, nothing to evaluate."
        
        if prop_vals is None: 
            prop_vals = self.cons_val
        if list_of_props is None: 
            list_of_props = self.prop_exp_list 
        ap_map = lambda x: list_of_props.index(x)

        # print(f"prop_vals = {prop_vals}")
        tl_value = evaluator(prop_vals, ap_map=ap_map, rho=rho)

        return tl_value


    def _gen_subform(self, con_groups: list, negate: bool = False, cnt_prop = 0) -> str:
        rule = ""
        prop_exp_list = []
        if len(con_groups) > 0:
            rule = "G "
            flg_hd_out = True
            for c_group in con_groups:
                if len(c_group) < 1:
                    continue 
                if not flg_hd_out: 
                    rule += " & "
                flg_hd_out = False
                rule += "~(" if negate else "("
                flg_hd = True
                for c in c_group:
                    cnt_prop +=1 
                    if not flg_hd:
                        rule += " & "
                    flg_hd = False
                    prop_exp = "c"+str(cnt_prop)
                    prop_exp_list.append(prop_exp)
                    rule += prop_exp
                    # print(cons_flat)
                    # cons_flat = c.eval()[None, :] if cons_flat is None else torch.concat((cons_flat, c.eval()[None, :]), dim=0)
                #end for
                rule += ")"
            #end for
        #end if
        return rule, prop_exp_list, cnt_prop


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


    def set_props_cons_and_ltl(self, con_groups, props, ltl):
        """
            con_groups: [func0, func1, func2, func3]
                each func should work on a 2-dim trajectory: Horizon x n_feature
            props: ['p0', 'p1', 'p2', 'p3']
            ltl: '(G~p0)'
        """
        assert len(con_groups) == len(props), "Number of constraint_groups and props do not match."
        self.n_cons_func = len(con_groups)
        self.cons_func_list = self._agg_cons_sub(con_groups)
        self.ap_map = lambda x: props.index(x)
        self.ltl = ltl
        self.formula_evaluator = parser.parse(self.ltl)


    def set_ltl_only(self, ltl):
        self.ltl = ltl
        self.formula_evaluator = parser.parse(self.ltl)
