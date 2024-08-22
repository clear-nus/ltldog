import numpy as np
import torch
import sys, os

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ..dtl_cont_cons import DTL_Cont_Cons_Evaluator


def test_2():
    st_np = np.array([
        np.array([
            [1., 2., 3., 4, 5],
            [1, 2, 3, 4, 5],
        ]).T,
        np.array([
            [7., 8., 9., 0, 2.7],
            [11, 20, -6.3, 88, -3.5],
        ]).T,
    ])
    st = torch.tensor(st_np.transpose((0,2,1)), dtype=torch.float, device="cuda", requires_grad=True)
    # st = st.transpose(1,2)
    print(f"state = {st}\n")

    evaluator = DTL_Cont_Cons_Evaluator()

    def f1(x):
        return 3*x[0] - 1
    def f2(x):
        return torch.square(x[0])+torch.square(x[1])-10
    def f3(x):
        return 4*x[1] - 103
    def f4(x):
        return torch.square(x[0])+torch.square(x[1])-0.99

    neg_cons = [[f2,f3]]
    pos_cons = [[f1], [f4]]

    print("Test generate_formula(): ")
    rule = evaluator.generate_formula(neg_cons, pos_cons)
    print(f"rule={rule}\n")

    print("Test aggregate_constraint(): ")
    evaluator.aggregate_constraint(neg_cons, pos_cons)
    print(f"cons_func_list = {evaluator.cons_func_list}\n")

    print("Test evaluate_constraints: ")
    cons_vals = evaluator.evaluate_constraints(st)
    print(f"cons_vals = {cons_vals}\n")

    print("Test formula_evaluation(): ")
    f_val = evaluator.formula_evaluation(st)
    print(f"formula_value = {f_val}\n")

    print("Test gradient backprop: ")
    f_val.backward(torch.ones_like(f_val))
    print(f"st.grad = {st.grad.transpose(1,2)}")


if __name__=='__main__':
    test_2()
