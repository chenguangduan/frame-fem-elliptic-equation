
import torch
import matplotlib.pyplot as plt
import numpy as np

from frame.pde import ParametricEllipticPDE
from frame.mesh import Mesh1D
from frame.fe import (_differential_frame_matrix_level, 
                      _diff_frame_scaled_per_level, 
                      _compute_sqrt_laplacian_diag_inv_per_level,
                      _assemble_parametric_diag,
                      _frame_analysis_fn,
                      _frame_synthesis_fn,
                      FiniteElementElliptic1D)

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
    return -1.0 + 0.0 * x

def main():
    num_levels = 10
    mesh = Mesh1D(n=2 ** num_levels, float_type=torch.float64, device=torch.device("cpu"))
    
    element_nodes_coords = mesh.nodes[mesh.elements]
    midpoint = 0.5 * (element_nodes_coords[:, 0] + element_nodes_coords[:, 1])
    mat = _differential_frame_matrix_level(
        idx_level=0,
        quadrature_points=midpoint,
        float_type=torch.float64, 
        device=torch.device("cpu")
    )
    #print(mat)

    sqrt_diag_inv_per_level = _compute_sqrt_laplacian_diag_inv_per_level(
        num_levels=num_levels,
        float_type=torch.float64, 
        device=torch.device("cpu")
    )
    #sqrt_diag_inv_per_level = torch.ones_like(sqrt_diag_inv_per_level)
    mat = _diff_frame_scaled_per_level(
        num_levels=num_levels,
        sqrt_diag_inv_per_level=sqrt_diag_inv_per_level,
        float_type=torch.float64, 
        device=torch.device("cpu")
    )
    print(mat)

    singular_values = torch.linalg.svdvals(mat)
    singular_values_max = singular_values[0]

    tol = torch.finfo(torch.float64).eps * max(mat.shape) * singular_values_max
    singular_values_nonzero = singular_values[singular_values > tol]
    singular_values_nonzero_min = singular_values_nonzero[-1]
    
    ratio = singular_values_max / singular_values_nonzero_min
    print(f"number level = {num_levels}, condition number = {ratio}")

    # --------------------------------------
    param = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64, device=torch.device("cpu"))
    param = param.unsqueeze(0)
    param_pde = ParametricEllipticPDE(
        diffusion_param_fn=diffusion_param_fn, 
        source_fn=source_fn
    )
    fe = FiniteElementElliptic1D(
        param_pde=param_pde, 
        num_levels=num_levels, 
        float_type=torch.float64, 
        device=torch.device("cpu"))

    diag = _assemble_parametric_diag(
        param=param,
        diffusion_param_fn=diffusion_param_fn,
        element_nodes_coords=mesh.nodes[mesh.elements]
    )
    frame_stiffness = mat.T @ (diag.T * mat)

    load = fe.assemble_load()
    frame_load = _frame_analysis_fn(
        num_levels=num_levels,
        sqrt_diag_inv_per_level=sqrt_diag_inv_per_level,
        vector_finest=load[:, 1:-1]
    ).squeeze(0)

    solution_frame_decomposition = torch.linalg.lstsq(frame_stiffness, frame_load, driver="gelss").solution
    solution_decomposition = _frame_synthesis_fn(
        num_levels=num_levels,
        sqrt_diag_inv_per_level=sqrt_diag_inv_per_level,
        coefficient_per_level=solution_frame_decomposition.unsqueeze(0)
    ).squeeze(0)

    #print(solution_decomposition)
    
    points = fe.nodes[1:-1]
    exact = 0.5 * points * (1.0 - points)

    print(torch.norm(solution_decomposition - exact))

    plt.plot(points, solution_decomposition.numpy(), 'r')
    plt.plot(points, exact, 'g')
    #plt.show()



if __name__ == "__main__":
    main()

