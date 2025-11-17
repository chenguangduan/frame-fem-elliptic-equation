
import torch
import matplotlib.pyplot as plt
import numpy as np

from frame.pde import ParametricEllipticPDE
from frame.mesh import Mesh1D
from frame.fe import FiniteElementElliptic1D, _diff_frame_scaled_vec_prod_per_level, _diff_frame_scaled_trans_vec_prod_per_level

def diffusion_param_fn(
    param: torch.Tensor, 
    x: torch.Tensor
    ) -> torch.Tensor:
    assert param.ndim == 2 and (x.ndim == 1 or x.ndim == 0)
    coef11 = (x >= 0.00) & (x < 0.25)
    coef12 = (x >= 0.25) & (x < 0.50)
    coef21 = (x >= 0.50) & (x < 0.75)
    coef22 = (x >= 0.75) & (x <= 1.00)
    output = param[:, 0:1] * coef11
    output += param[:, 1:2] * coef12
    output += param[:, 2:3] * coef21
    output += param[:, 3:4] * coef22
    return output 

def source_fn(x: torch.Tensor):
    return 1.0 + 0.0 * x

def compute_condition_number(mat):
    singular_values = torch.linalg.svdvals(mat)
    singular_values_max = singular_values[0]

    float_type = mat.dtype
    tol = torch.finfo(float_type).eps * max(mat.shape) * singular_values_max
    singular_values_nonzero = singular_values[singular_values > tol]
    singular_values_nonzero_min = singular_values_nonzero[-1]
    
    cond = singular_values_max / singular_values_nonzero_min
    return cond

def main():
    float_type = torch.float64
    device = torch.device("cpu")

    param = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=float_type, device=device)
    param = param.unsqueeze(0)

    param_pde = ParametricEllipticPDE(
        diffusion_param_fn=diffusion_param_fn, 
        source_fn=source_fn
    )

    num_levels = 10
    fe = FiniteElementElliptic1D(
        param_pde=param_pde, 
        num_levels=num_levels, 
        float_type=float_type, 
        device=device)
    
    diff_frame_scaled_per_level = fe.diff_frame_scaled_per_level
    cond = compute_condition_number(diff_frame_scaled_per_level)
    print(f"number of levels = {num_levels}, condition number = {cond}")

    num_coefficients = fe.num_coefficients
    test_coefficient = torch.rand(1, num_coefficients, dtype=float_type, device=device)
    # decomposition
    output_decomposition_mat = fe.assemble_frame_mat_coef_prod_decomposition_mat(
        param=param,
        input_coefficient=test_coefficient
    )
    # decomposition
    output_decomposition = fe.assemble_frame_mat_coef_prod_decomposition(
        param=param,
        input_coefficient=test_coefficient
    )
    # original
    output = fe.assemble_frame_mat_coef_prod(
        param=param,
        input_coefficient=test_coefficient
    )
    print(f"norm without decomposition = {torch.norm(output).item():>.4e}")
    print(f"norm with decomposition = {torch.norm(output_decomposition).item():>.4e}")
    print(f"norm with decomposition = {torch.norm(output_decomposition_mat).item():>.4e}")
    print(f"error = {torch.norm(output - output_decomposition).item():>.4e}")
    print(f"error = {torch.norm(output - output_decomposition_mat).item():>.4e}")
    print(f"error = {torch.norm(output_decomposition - output_decomposition_mat).item():>.4e}")


if __name__ == "__main__":
    main()
