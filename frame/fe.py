import abc
from typing import List, Tuple, Callable, cast, Optional

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

import torch
from torch.nn import functional as F

from .pde import ParametricEllipticPDE
from .mesh import Mesh1D


class FiniteElementElliptic1D(abc.ABC):
    def __init__(
            self, 
            param_pde: ParametricEllipticPDE, 
            num_levels: int,
            float_type: torch.dtype,
            device: torch.device,
            ) -> None:
        # Define the parametric PDE
        self.param_pde: ParametricEllipticPDE = param_pde
        # Define the float type
        self.float_type: torch.dtype = float_type
        # Define the device
        self.device: torch.device = device
        # Generate the mesh
        self.num_levels = num_levels
        self.num_nodes = 2 ** num_levels + 1
        mesh = Mesh1D(n=2 ** num_levels, float_type=self.float_type, device=self.device)
        self.nodes: torch.Tensor = mesh.nodes
        self.elements: torch.Tensor = mesh.elements
        self.num_nodes: int = self.nodes.size(0)
        self.num_elements: int = self.elements.size(0)
        # Degrees of freedom (DoFs)
        self.num_dofs: int =  self.num_nodes
        self.dofs = torch.arange(self.num_dofs, dtype=torch.int64, device=self.device)
        self.dirichlet_dofs = self.dofs[mesh.dirichlet_mask]
        self.free_dofs_mask = torch.ones(self.num_dofs, dtype=torch.bool, device=self.device)
        self.free_dofs_mask[self.dirichlet_dofs] = False
        self.free_dofs = self.dofs[self.free_dofs_mask]
        self.num_free_dofs = self.free_dofs.size(0)
        # Preconditioning
        self.num_coefficients = sum(2 ** (k + 1) - 1 for k in range(num_levels))
        self.sqrt_diag_inv_per_level = _compute_sqrt_laplacian_diag_inv_per_level(
            num_levels=self.num_levels,
            float_type=self.float_type,
            device=self.device
        )
        self.diff_frame_scaled_per_level = _diff_frame_scaled_per_level(
            num_levels=num_levels,
            sqrt_diag_inv_per_level=self.sqrt_diag_inv_per_level,
            float_type=self.float_type,
            device=self.device
        )
        # decomposition
        element_nodes_coords = mesh.nodes[mesh.elements]
        quadrature_points = 0.5 * (element_nodes_coords[:, 0] + element_nodes_coords[:, 1])
        self.index_elements_per_level = _compute_quadrature_points_element_index_per_level(
            num_levels=self.num_levels,
            quadrature_points=quadrature_points
        )
        self.mesh_per_level: Tuple[Mesh1D, ...] = tuple(Mesh1D(n=2 ** (idx_level + 1), float_type=self.float_type, device=self.device)
                          for idx_level in range(self.num_levels))
        

    def to(self, device: torch.device):
        device = torch.device(device)
        # List of tensor attributes to be moved
        tensor_attributes_name = [
            "nodes", "elements", "dofs", "dirichlet_dofs", 
            "free_dofs_mask", "free_dofs", "sqrt_diag_inv_per_level",
            "one_side_grad_frame", "one_side_div_frame"
        ]
        for attribute_name in tensor_attributes_name:
            tensor = getattr(self, attribute_name)
            setattr(self, attribute_name, tensor.to(device))
        self.device = device 
        return self
    
    def cpu(self):
        return self.to(torch.device("cpu"))
    
    def double(self):
        # List of tensor attributes
        float_tensor_attributes_name = [
            "nodes", "sqrt_diag_inv_per_level",
            "one_side_grad_frame", "one_side_div_frame"
        ]
        for attribute_name in float_tensor_attributes_name:
            float_tensor = getattr(self, attribute_name)
            setattr(self, attribute_name, float_tensor.double())
        self.float_type = torch.float64 
        return self
    
    def float(self):
        # List of tensor attributes
        float_tensor_attributes_name = [
            "nodes", "sqrt_diag_inv_per_level",
            "one_side_grad_frame", "one_side_div_frame"
        ]
        for attribute_name in float_tensor_attributes_name:
            float_tensor = getattr(self, attribute_name)
            setattr(self, attribute_name, float_tensor.float())
        self.float_type = torch.float32
        return self
    
    def half(self):
        # List of tensor attributes
        float_tensor_attributes_name = [
            "nodes", "sqrt_diag_inv_per_level",
            "one_side_grad_frame", "one_side_div_frame"
        ]
        for attribute_name in float_tensor_attributes_name:
            float_tensor = getattr(self, attribute_name)
            setattr(self, attribute_name, float_tensor.half())
        self.float_type = torch.float16
        return self
    
    def assemble_load(self) -> torch.Tensor:
        source_fn = self.param_pde.get_source_fn()
        # load_vec.shape = (num_dofs)
        load_vec = _assemble_load(
            source_fn=source_fn, 
            nodes=self.nodes, 
            elements=self.elements)
        # shape of output = (1, num_dofs)
        return load_vec[None, :]
    
    def extend_solution_batch(
            self, 
            solution_reduced: torch.Tensor,
            ) -> torch.Tensor:
        # solution_reduced.shape = (batch_size, num_free_dofs)
        # solution.shape = (batch_size, num_dofs)
        device = solution_reduced.device
        assert solution_reduced.ndim == 2
        batch_size: int = solution_reduced.size(0)
        # Homogenous Dirichlet boundar condition
        # solution[:, dirichlet_dofs] = 0.0
        # solution[:, free_dofs] = solution_reduced
        solution = torch.zeros(
            batch_size, self.num_dofs, dtype=self.float_type).to(device)
        solution[:, self.free_dofs] = solution_reduced
        return solution
    
    # -------------------------------------------------
    def frame_analysis(
            self, 
            param: torch.Tensor, 
            vector_finest: torch.Tensor
            ) -> torch.Tensor:
        # param.shape = (batch_size, dim(param))
        # vector_finest.shape = (batch_size, num_free_dofs)
        assert param.ndim == 2 and vector_finest.ndim == 2
        assert vector_finest.size(1) == self.num_free_dofs
        # From the finest grid to coefficients in each level
        # coefficient_per_level.shape = (batch_size, num_coefficients)
        coefficient_per_level = _frame_analysis_fn(
            num_levels=self.num_levels, 
            sqrt_diag_inv_per_level=self.sqrt_diag_inv_per_level,
            vector_finest=vector_finest
        )
        return coefficient_per_level
    
    def frame_synthesis(
            self, 
            param: torch.Tensor, 
            coefficient_per_level: torch.Tensor
            ) -> torch.Tensor:
        # param.shape = (batch_size, dim(param))
        # coefficient_per_level.shape = (batch_size, num_coefficients)
        assert param.ndim == 2 and coefficient_per_level.ndim == 2
        assert coefficient_per_level.size(1) == self.num_coefficients
        # From the coefficients in each level to the finest grid
        # vector_finest.shape = (batch_size, num_free_dofs)
        vector_finest = _frame_synthesis_fn(
            num_levels=self.num_levels, 
            sqrt_diag_inv_per_level=self.sqrt_diag_inv_per_level,
            coefficient_per_level=coefficient_per_level
        )
        return vector_finest
    
    def assemble_frame_mat_coef_prod_decomposition(
            self, 
            param: torch.Tensor,
            input_coefficient: torch.Tensor,
            ):
        diff_frame_prod = _diff_frame_scaled_vec_prod_per_level(
            mesh_per_level=self.mesh_per_level,
            index_elements_per_level=self.index_elements_per_level,
            sqrt_diag_inv_per_level=self.sqrt_diag_inv_per_level,
            input_coefficients=input_coefficient
        )
        param_diag = _assemble_parametric_diag(
            param=param,
            diffusion_param_fn=self.param_pde.diffusion_param_fn,
            element_nodes_coords=self.nodes[self.elements]
        )
        return _diff_frame_scaled_trans_vec_prod_per_level(
            mesh_per_level=self.mesh_per_level,
            index_elements_per_level=self.index_elements_per_level,
            sqrt_diag_inv_per_level=self.sqrt_diag_inv_per_level,
            input_vec=diff_frame_prod * param_diag
        )
    
    def assemble_frame_mat_coef_prod_decomposition_mat(
            self, 
            param: torch.Tensor,
            input_coefficient: torch.Tensor,
            ):
        param_diag = _assemble_parametric_diag(
            param=param,
            diffusion_param_fn=self.param_pde.diffusion_param_fn,
            element_nodes_coords=self.nodes[self.elements]
        )
        diff_frame_scaled_coeff_prod = torch.einsum(
            "ec,bc->be", 
            self.diff_frame_scaled_per_level,
            input_coefficient)
        output = torch.einsum(
            "ec,be->bc",
            self.diff_frame_scaled_per_level,
            param_diag * diff_frame_scaled_coeff_prod
        )
        return output
    
    def assemble_frame_mat_coef_prod(
            self, 
            param: torch.Tensor,
            input_coefficient: torch.Tensor,
            ) -> torch.Tensor:
        # param.shape = (batch_size, dim(param))
        # input_coefficient.shape = (batch_size, num_coefficients)
        assert param.ndim == 2 and input_coefficient.ndim == 2
        assert input_coefficient.size(1) == self.num_coefficients
        # Get the parametric diffusion function of the PDE
        diffusion_param_fn = self.param_pde.get_diffusion_param_fn()
        # From the coefficients in each level to the finest grid
        # vector_finest_reduced.shape = (batch_size, num_free_dofs)
        vector_finest_reduced = self.frame_synthesis(
            param=param, coefficient_per_level=input_coefficient)
        # cat vector_finest_reduced and homogenous Dirichlet boundary condition
        # vector_finest.shape = (batch_size, num_dofs)
        vector_finest = self.extend_solution_batch(
            solution_reduced=vector_finest_reduced)
        # Compute the matrix-vector product
        # mat_vec_prod.shape = (batch_size, num_dofs)
        mat_vec_prod = _assemble_mat_vec_prod(
            param=param, 
            diffusion_param_fn=diffusion_param_fn, 
            input_vec=vector_finest, 
            nodes=self.nodes, 
            elements=self.elements)
        # Remove Dirichlet DoFs
        # mat_vec_prod_reduced.shape = (batch_size, num_free_dofs)
        mat_vec_prod_reduced = mat_vec_prod[:, self.free_dofs]
        # From the finest grid to coefficients in each level
        # output shape = (batch_size, num_coefficients)
        return self.frame_analysis(
            param=param, 
            vector_finest=mat_vec_prod_reduced)
    
    def assemble_frame_load(self, param: torch.Tensor) -> torch.Tensor:
        # param.shape = (batch_size, dim(param))
        device = param.device
        assert param.ndim == 2
        batch_size = param.size(0)
        # Get the source function of the PDE
        source_fn = self.param_pde.get_source_fn()
        # Get the load vector
        # load_vec.shape = (1, num_dofs)
        load_vec = _assemble_load(
            source_fn=source_fn, nodes=self.nodes, elements=self.elements)
        # Remove the dirichlet_dofs, and expand the first dimension
        # load_vec.shape = (batch_size, num_free_dofs)
        load_vec = load_vec[None, self.free_dofs].expand(batch_size, -1).to(device)
        # From the finest grid to the coefficients in each level
        # frame_load.shape = (batch_size, num_coefficients)
        frame_load = self.frame_analysis(
            param=param, 
            vector_finest=load_vec)
        return frame_load

    # -------------------------------------------------
    def _extend_solution(
            self, 
            solution_reduced: torch.Tensor,
            ) -> torch.Tensor:
        # solution_reduced.shape = (num_free_dofs)
        # solution.shape = (num_dofs)
        assert solution_reduced.ndim == 1
        # Homogenous Dirichlet boundar condition
        # solution[dirichlet_dofs] = 0
        # solution[free_dofs] = solution_reduced
        solution = torch.zeros(self.num_dofs, dtype=self.float_type)
        solution[self.free_dofs] = solution_reduced
        return solution
    
    def _assemble_matrix_reduced(
            self, 
            param: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        # param.shape = (dim(param))
        assert param.ndim == 1
        diffusion_param_fn = self.param_pde.get_diffusion_param_fn()
        # matrix_reduced.shape = (num_free_dofs, num_free_dofs)
        matrix_reduced, matrix = _assemble(param=param, diffusion_param_fn=diffusion_param_fn, 
                                   nodes=self.nodes, elements=self.elements, 
                                   free_dofs=self.free_dofs)
        return matrix_reduced, matrix
    
    def _assemble_load_reduced(self) -> torch.Tensor:
        source_fn = self.param_pde.get_source_fn()
        # load_vec.shape = (num_dofs)
        load_vec = _assemble_load(
            source_fn=source_fn, 
            nodes=self.nodes, 
            elements=self.elements)
        # shape of output = (num_dofs)
        return load_vec[self.free_dofs]

    def solve(
            self, 
            param: torch.Tensor
            ) -> Tuple[npt.NDArray, npt.NDArray]:
        # param.shape = (dim(param))
        assert param.ndim == 1
        # Assemble the reduced matrix, torch.sparse_csr_tensor
        # K_torch_csr.shape = (num_free_dofs, num_free_dofs)
        K_torch_csr = self._assemble_matrix_reduced(param=param)[0]
        F_torch = self._assemble_load_reduced()
        solution_reduced = _sparse_solve_torch(K_torch_csr, F_torch)
        solution = self._extend_solution(torch.from_numpy(solution_reduced)).numpy()
        return solution_reduced, solution
    

# ------------------- Decomposition ------------------

def _assemble_load(
        source_fn: Callable,
        nodes: torch.Tensor, 
        elements: torch.Tensor,
        ) -> torch.Tensor:
    device: torch.device = nodes.device
    float_type: torch.dtype = nodes.dtype
    num_nodes: int = nodes.size(0)
    num_dofs: int =  num_nodes
    # F_global.shape = (num_dofs)
    F_global = torch.zeros(num_dofs, dtype=float_type, device=device)
    # local_vector.shape = (num_elements, 2)
    local_vector = _compute_local_vector(source_fn=source_fn, element_nodes_coords=nodes[elements])
    # global_dofs.shape = (num_elements, 2)
    global_dofs = elements
    # scatter
    src = local_vector.reshape(-1)
    index = global_dofs.reshape(-1)
    F_global.scatter_add_(dim=0, index=index, src=src)
    return F_global

def _prolongation_fn(
        vector_coarse: torch.Tensor
        ) -> torch.Tensor:
    # vector_coarse.shape = (batch_size, vector_corase dim)
    assert vector_coarse.ndim == 2
    device = vector_coarse.device
    float_type = vector_coarse.dtype
    kernel = torch.tensor([0.5, 1.0, 0.5], dtype=float_type, device=device).view(1, 1, 3)
    vector_finer = F.conv_transpose1d(vector_coarse.unsqueeze(1), kernel, stride=2, padding=0)
    return vector_finer.squeeze(1)
    
def _restriction_fn(
        vector_fine: torch.Tensor
        ) -> torch.Tensor:
    # vector_fine.shape = (batch_size, vector_fine dim)
    assert vector_fine.ndim == 2
    device = vector_fine.device
    float_type = vector_fine.dtype
    kernel = torch.tensor([0.5, 1.0, 0.5], dtype=float_type, device=device).view(1, 1, 3)
    vecotr_coarser = F.conv1d(vector_fine.unsqueeze(1), kernel, stride=2, padding=0)
    return vecotr_coarser.squeeze(1)

def _frame_synthesis_fn(
        num_levels: int,
        sqrt_diag_inv_per_level: torch.Tensor,
        coefficient_per_level: torch.Tensor,
        ) -> torch.Tensor:
    # sqrt_diag_inv_per_level.shape = (num_levels, num_nodes_finest)
    assert coefficient_per_level.ndim == 2
    # coefficient_per_level.shape = (batch_size, num_coefficients,)
    # Run the lean, sequential synthesis loop
    coeff_start_idx = 0
    num_nodes_current_level = 1
    coeff_end_idx = coeff_start_idx + num_nodes_current_level
    sqrt_diag_current = sqrt_diag_inv_per_level[0, :num_nodes_current_level]
    coefficient_current = coefficient_per_level[:, coeff_start_idx:coeff_end_idx]
    vector_current_level = sqrt_diag_current * coefficient_current
    coeff_start_idx = coeff_end_idx
    for idx_level in range(1, num_levels):
        Pmat_vec_prod = _prolongation_fn(vector_coarse=vector_current_level)
        num_nodes_current_level = 2 ** (idx_level + 1) - 1
        coeff_end_idx = coeff_start_idx + num_nodes_current_level
        sqrt_diag_current = sqrt_diag_inv_per_level[idx_level, :num_nodes_current_level]
        coefficient_current = coefficient_per_level[:, coeff_start_idx:coeff_end_idx]
        vector_current_level = Pmat_vec_prod + sqrt_diag_current * coefficient_current
        coeff_start_idx = coeff_end_idx
    return vector_current_level

def _frame_analysis_fn(
        num_levels: int,
        sqrt_diag_inv_per_level: torch.Tensor,
        vector_finest: torch.Tensor,
        ) -> torch.Tensor:
    # sqrt_diag_inv_per_level.shape = (num_levels, num_nodes_finest)
    assert vector_finest.ndim == 2
    # vector_finest.shape = (batch_size, num_free_dofs)
    device = vector_finest.device
    float_type = vector_finest.dtype
    batch_size = vector_finest.size(0)
    # Run the lean, sequential analysis loop
    vector_current_level = vector_finest
    num_coefficients = sum(2 ** (k + 1) - 1 for k in range(num_levels))
    coefficient_per_level = torch.zeros(
        batch_size, num_coefficients, dtype=float_type, device=device)
    coeff_end_idx = num_coefficients
    for idx_level in reversed(range(1, num_levels)):
        num_nodes_current_level = 2 ** (idx_level + 1) - 1
        coeff_start_idx = coeff_end_idx - num_nodes_current_level
        sqrt_diag_current = sqrt_diag_inv_per_level[idx_level, :num_nodes_current_level]
        coefficient_current = sqrt_diag_current * vector_current_level
        coefficient_per_level[:, coeff_start_idx:coeff_end_idx] = coefficient_current
        coeff_end_idx = coeff_start_idx
        vector_current_level = _restriction_fn(vector_fine=vector_current_level)
    # coarsest level
    num_nodes_current_level = 1
    sqrt_diag_current = sqrt_diag_inv_per_level[0, :num_nodes_current_level]
    coefficient_current = sqrt_diag_current * vector_current_level
    coefficient_per_level[:, 0:coeff_end_idx] = coefficient_current
    return coefficient_per_level

def _compute_local_laplacian_diag(
        element_nodes_coords: torch.Tensor,
        ):
    # param.shape = (batch_size, param_dim)
    # Compute the length and midpoint of the elements
    # h.shape = midpoint.shape = (num_elements,)
    h = torch.abs(element_nodes_coords[:, 0] - element_nodes_coords[:, 1])
    # Basis functions and their derivatives
    # dNdx.shape = (num_elements, 2)
    dNdx = torch.cat((-1.0 / h[:, None], 1.0 / h[:, None]), dim=1)
    # dNdxdNdx_diag: (1, num_elements, 2)
    dNdxdNdx_diag = (dNdx * dNdx)[None, :, :]
    # Assemble the local diagonal matrix
    # local_laplacian_diag.shape = (num_elements, 2)
    local_laplacian_diag = dNdxdNdx_diag * h[None, :, None]
    return local_laplacian_diag

def _assemble_laplacian_diag(
        nodes: torch.Tensor, 
        elements: torch.Tensor,
        free_dofs: torch.Tensor,
        ) -> torch.Tensor:
    # param.shape = (batch_size, param_dim)
    device: torch.device = nodes.device
    float_type: torch.dtype = nodes.dtype
    # The number of nodes and number of degrees of freedom
    num_nodes: int = nodes.size(0)
    num_dofs: int =  num_nodes
    # Initialize 
    laplacian_diag = torch.zeros(num_dofs, dtype=float_type, device=device)
    # local_laplacian_diag.shape = (num_elements, 2)
    local_laplacian_diag = _compute_local_laplacian_diag(element_nodes_coords=nodes[elements])
    # global_dofs.shape = (num_elements, 2)
    global_dofs = elements
    # scatter
    src = local_laplacian_diag.reshape(-1)
    index = global_dofs.reshape(-1)
    laplacian_diag.scatter_add_(dim=0, index=index, src=src)
    # return shape = (num_free_dofs,)
    return laplacian_diag[free_dofs]

def _assemble_laplacian_diag_level(
        idx_level: int,
        float_type: torch.dtype,
        device: torch.device,
        ):
    # Generate the mesh for this level
    mesh = Mesh1D(n=2 ** (idx_level + 1), float_type=float_type, device=device)
    num_nodes = mesh.num_nodes
    num_dofs = num_nodes
    dofs = torch.arange(num_dofs, dtype=torch.int64, device=device)
    free_dofs_mask = torch.ones(num_dofs, dtype=torch.bool, device=device)
    free_dofs_mask[dofs[mesh.dirichlet_mask]] = False
    free_dofs = dofs[free_dofs_mask]
    # Compute the diagonal matrix 
    laplacian_diag = _assemble_laplacian_diag(
        nodes=mesh.nodes, 
        elements=mesh.elements,
        free_dofs=free_dofs)
    # Compute the inverse of the diagonal matrix
    return laplacian_diag

def _compute_sqrt_laplacian_diag_inv_per_level(
        num_levels: int,
        float_type: torch.dtype,
        device: torch.device,
        ) -> torch.Tensor:
    num_free_dofs_finest = 2 ** num_levels - 1
    sqrt_laplacian_diag_inv = torch.zeros(num_levels, num_free_dofs_finest, dtype=float_type, device=device)
    for idx_level in range(num_levels):
        num_free_dofs_current_level = 2 ** (idx_level + 1) - 1
        laplacian_diag = _assemble_laplacian_diag_level(idx_level=idx_level, float_type=float_type, device=device)
        sqrt_laplacian_diag_inv[idx_level, :num_free_dofs_current_level] = torch.sqrt(1.0 / laplacian_diag)
    return sqrt_laplacian_diag_inv

def _assemble_parametric_diag(
        param: torch.Tensor,
        diffusion_param_fn: Callable,
        element_nodes_coords: torch.Tensor,
        ) -> torch.Tensor:
    # param.shape = (batch_size, param_dim)
    # h.shape = midpoint.shape = (num_elements,)
    h = torch.abs(element_nodes_coords[:, 0] - element_nodes_coords[:, 1])
    midpoint = 0.5 * (element_nodes_coords[:, 0] + element_nodes_coords[:, 1])
    # Diffusion function evaluation at the midpoint of each element
    # diffusion_mid.shape = (batch_size, num_elements)
    diffusion_mid = diffusion_param_fn(param, midpoint)
    # parametric_diag.shape = (batch_size, num_elements)
    parametric_diag = diffusion_mid * h[None, :]
    return parametric_diag

def _compute_local_differential(
        element_nodes_coords: torch.Tensor,
        ) -> torch.Tensor:
    # Compute the length and midpoint of the elements
    # h.shape = midpoint.shape = (num_elements,)
    h = torch.abs(element_nodes_coords[:, 0] - element_nodes_coords[:, 1])
    # dNdx.shape = (num_elements, 2)
    dNdx = torch.cat((-1.0 / h[:, None], 1.0 / h[:, None]), dim=1)
    return dNdx

def _assemble_differential_matrix(
        nodes,
        elements,
        free_dofs,
        ) -> torch.Tensor:
    device: torch.device = nodes.device
    float_type: torch.dtype = nodes.dtype
    num_elements = elements.size(0)
    num_dofs = nodes.size(0)
    # h.shape = midpoint.shape = (num_elements,)
    element_nodes_coords = nodes[elements]
    local_differential = _compute_local_differential(
        element_nodes_coords=element_nodes_coords
    )
    # mat.shape = (num_elements, num_dofs)
    mat = torch.zeros(num_elements, num_dofs, dtype=float_type, device=device)
    for idx_element, element in enumerate(elements):
        mat[idx_element, element] = local_differential[idx_element]
    return mat[:, free_dofs]

def _diff_frame_level(
        idx_level: int,
        quadrature_points: torch.Tensor,
        float_type: torch.dtype,
        device: torch.device,
        ) -> torch.Tensor:
    # Generate the mesh for this level
    mesh = Mesh1D(n=2 ** (idx_level + 1), float_type=float_type, device=device)
    nodes = mesh.nodes 
    elements = mesh.elements
    num_dofs = nodes.size(0)
    dofs = torch.arange(num_dofs, dtype=torch.int64, device=device)
    dirichlet_dofs = dofs[mesh.dirichlet_mask]
    free_dofs_mask = torch.ones(num_dofs, dtype=torch.bool, device=device)
    free_dofs_mask[dirichlet_dofs] = False
    free_dofs = dofs[free_dofs_mask]
    # element_nodes_coords.shape = (num_elements, 2)
    element_nodes_coords = nodes[elements]
    end_point_min = torch.min(input=element_nodes_coords, dim=1, keepdim=True)[0]
    end_point_max = torch.max(input=element_nodes_coords, dim=1, keepdim=True)[0]
    # quadrature_points.shape = (num_points, 1)
    # index_mask.shape = (num_points, num_elements)
    quadrature_points = quadrature_points.unsqueeze(-1)
    index_mask = (quadrature_points >= end_point_min.T) & (quadrature_points < end_point_max.T)
    index_element = torch.arange(elements.size(0)).expand(quadrature_points.size(0), -1)
    # index_element.shape = (num_points,)
    index_element = index_element[index_mask]
    # mat.shape = (num_elements, num_free_dofs)
    mat = _assemble_differential_matrix(nodes=nodes, elements=elements, free_dofs=free_dofs)
    return mat[index_element, :]

def _diff_frame_scaled_per_level(
        num_levels: int,
        sqrt_diag_inv_per_level: torch.Tensor,
        float_type: torch.dtype,
        device: torch.device,
        ) -> torch.Tensor:
    mesh = Mesh1D(n=2 ** num_levels, float_type=float_type, device=device)
    element_nodes_coords = mesh.nodes[mesh.elements]
    midpoint = 0.5 * (element_nodes_coords[:, 0] + element_nodes_coords[:, 1])
    num_coefficients = sum(2 ** (k + 1) - 1 for k in range(num_levels))
    scaled_mat = torch.zeros(midpoint.size(0), num_coefficients, dtype=float_type, device=device)
    # loop
    coeff_start_idx = 0
    for idx_level in range(num_levels):
        num_nodes_current_level = 2 ** (idx_level + 1) - 1
        coeff_end_idx = coeff_start_idx + num_nodes_current_level
        sqrt_diag_inv_current = sqrt_diag_inv_per_level[idx_level, :num_nodes_current_level]
        mat = _diff_frame_level(
            idx_level=idx_level, 
            quadrature_points=midpoint,
            float_type=float_type,
            device=device)
        scaled_mat[:, coeff_start_idx:coeff_end_idx] = mat * sqrt_diag_inv_current
        coeff_start_idx = coeff_end_idx
    return scaled_mat

def _compute_quadrature_points_element_index_level(
        idx_level: int,
        quadrature_points: torch.Tensor,
        ):
    float_type = quadrature_points.dtype
    device = quadrature_points.device
    # Generate the mesh for this level
    mesh = Mesh1D(n=2 ** (idx_level + 1), float_type=float_type, device=device)
    nodes = mesh.nodes 
    elements = mesh.elements
    # element_nodes_coords.shape = (num_elements, 2)
    element_nodes_coords = nodes[elements]
    end_point_min = torch.min(input=element_nodes_coords, dim=1, keepdim=True)[0]
    end_point_max = torch.max(input=element_nodes_coords, dim=1, keepdim=True)[0]
    # quadrature_points.shape = (num_points, 1)
    # index_mask.shape = (num_points, num_elements)
    quadrature_points = quadrature_points.unsqueeze(-1)
    index_mask = (quadrature_points >= end_point_min.T) & (quadrature_points < end_point_max.T)
    index_element = torch.arange(elements.size(0)).expand(quadrature_points.size(0), -1)
    # index_element.shape = (num_points,)
    index_element = index_element[index_mask]
    return index_element

def _compute_quadrature_points_element_index_per_level(
        num_levels: int,
        quadrature_points: torch.Tensor,
        ) -> torch.Tensor:
    device = quadrature_points.device
    num_points = quadrature_points.size(0)
    index_element_per_level = torch.zeros(num_points, num_levels, dtype=torch.int64, device=device)
    for idx_level in range(num_levels):
        index_element_per_level[:, idx_level] = _compute_quadrature_points_element_index_level(
            idx_level=idx_level,
            quadrature_points=quadrature_points,
        )
    return index_element_per_level

def _diff_frame_scaled_vec_prod_level(
        mesh: Mesh1D,
        index_elements_level: torch.Tensor,
        sqrt_diag_inv_level: torch.Tensor,
        input_vec: torch.Tensor,
        ) -> torch.Tensor:
    float_type = input_vec.dtype
    device = input_vec.device
    nodes = mesh.nodes 
    elements = mesh.elements
    free_dofs = mesh.free_dofs
    # 
    batch_size = input_vec.size(0)
    num_free_dofs = input_vec.size(1)
    input_vec_scaled_reduced = input_vec * sqrt_diag_inv_level.unsqueeze(0)
    input_vec_scaled = torch.zeros(batch_size, num_free_dofs + 2, dtype=float_type, device=device)
    input_vec_scaled[:, free_dofs] = input_vec_scaled_reduced
    #
    local_diff = _compute_local_differential(element_nodes_coords=nodes[elements])
    # local_diff_quadrature_points.shape = (num_points, 2)
    local_diff_quadrature_points = local_diff[index_elements_level]
    # global_dofs.shape = (num_points, 2)
    global_dofs = elements[index_elements_level]
    # local_vec.shape = (batch_size, num_points, 2)
    local_vec_scaled = input_vec_scaled[:, global_dofs]
    # product
    prod = torch.einsum("pi,bpi->bp", local_diff_quadrature_points, local_vec_scaled)
    return prod 


def _diff_frame_scaled_vec_prod_per_level(
        mesh_per_level: Tuple[Mesh1D, ...],
        index_elements_per_level: torch.Tensor,
        sqrt_diag_inv_per_level: torch.Tensor,
        input_coefficients: torch.Tensor,
        ) -> torch.Tensor:
    float_type = input_coefficients.dtype
    device = input_coefficients.device

    num_levels = len(mesh_per_level)
    batch_size = input_coefficients.size(0)
    num_points = index_elements_per_level.size(0)
    prod = torch.zeros(batch_size, num_points, dtype=float_type, device=device)

    coeff_start_idx = 0
    for idx_level in range(num_levels):
        num_nodes_current_level = 2 ** (idx_level + 1) - 1
        coeff_end_idx = coeff_start_idx + num_nodes_current_level
        sqrt_diag_inv_current = sqrt_diag_inv_per_level[idx_level, :num_nodes_current_level]
        prod += _diff_frame_scaled_vec_prod_level(
            mesh=mesh_per_level[idx_level],
            index_elements_level=index_elements_per_level[:, idx_level],
            sqrt_diag_inv_level=sqrt_diag_inv_current,
            input_vec=input_coefficients[:, coeff_start_idx:coeff_end_idx],
        )
        coeff_start_idx = coeff_end_idx
    return prod


def _diff_frame_scaled_trans_vec_prod_level(
        mesh: Mesh1D,
        index_elements_level: torch.Tensor,
        sqrt_diag_inv_level: torch.Tensor,
        input_vec: torch.Tensor,
        ) -> torch.Tensor:
    float_type = input_vec.dtype
    device = input_vec.device
    nodes = mesh.nodes 
    elements = mesh.elements
    free_dofs = mesh.free_dofs
    batch_size = input_vec.size(0)
    num_nodes = nodes.size(0)
    #
    local_diff = _compute_local_differential(element_nodes_coords=nodes[elements])
    # local_diff_quadrature_points.shape = (num_points, 2)
    local_diff_quadrature_points = local_diff[index_elements_level]
    # global_dofs.shape = (num_points, 2)
    global_dofs = elements[index_elements_level]

    # scatter 
    # output_vec.shape = (batch_size, num_nodes)
    src = local_diff_quadrature_points.unsqueeze(0) * input_vec.unsqueeze(-1)
    src = src.reshape(batch_size, -1)
    index = global_dofs.reshape(1, -1).expand(batch_size, -1)
    output_vec = torch.zeros(batch_size, num_nodes, dtype=float_type, device=device)
    output_vec.scatter_add_(dim=1, index=index, src=src)
    # scale
    return output_vec[:, free_dofs] * sqrt_diag_inv_level.unsqueeze(0)
 

def _diff_frame_scaled_trans_vec_prod_per_level(
        mesh_per_level: Tuple[Mesh1D, ...],
        index_elements_per_level: torch.Tensor,
        sqrt_diag_inv_per_level: torch.Tensor,
        input_vec: torch.Tensor,
        ) -> torch.Tensor:
    float_type = input_vec.dtype
    device = input_vec.device

    num_levels = len(mesh_per_level)
    batch_size = input_vec.size(0)
    num_coefficients = sum(2 ** (k + 1) - 1 for k in range(num_levels))
    prod = torch.zeros(batch_size, num_coefficients, dtype=float_type, device=device)

    coeff_start_idx = 0
    for idx_level in range(num_levels):
        num_nodes_current_level = 2 ** (idx_level + 1) - 1
        coeff_end_idx = coeff_start_idx + num_nodes_current_level
        sqrt_diag_inv_current = sqrt_diag_inv_per_level[idx_level, :num_nodes_current_level]
        prod[:, coeff_start_idx:coeff_end_idx] = _diff_frame_scaled_trans_vec_prod_level(
            mesh=mesh_per_level[idx_level],
            index_elements_level=index_elements_per_level[:, idx_level],
            sqrt_diag_inv_level=sqrt_diag_inv_current,
            input_vec=input_vec,
        )
        coeff_start_idx = coeff_end_idx
    return prod


# ------------------- Local matrix and local laod vector ------------- 

def _compute_local_matrix(
        param: torch.Tensor,
        diffusion_param_fn,
        element_nodes_coords: torch.Tensor,
        ) -> torch.Tensor:
    # param.shape = (batch_size, param_dim)
    assert param.ndim == 2
    
    # Compute the length and midpoint of the elements
    # h.shape = midpoint.shape = (num_elements,)
    h = torch.abs(element_nodes_coords[:, 0] - element_nodes_coords[:, 1])
    midpoint = 0.5 * (element_nodes_coords[:, 0] + element_nodes_coords[:, 1])
    
    # Diffusion function evaluation at the midpoint of each element
    # diffusion_inv_mid.shape = (batch_size, num_elements, 1, 1)
    diffusion_mid = diffusion_param_fn(param, midpoint)[:, :, None, None]
    
    # dNdx.shape = (num_elements, 2)
    dNdx = torch.cat((-1.0 / h[:, None], 1.0 / h[:, None]), dim=1)
    
    # Outer products for better GPU utilization
    # dNdxdNdx.shape = (1, num_elements, 2, 2)
    dNdxdNdx = torch.einsum("ei,ej->eij", dNdx, dNdx)[None, :, :, :]
    
    # Assemble matrix blocks with vectorized operations
    # local_matrix.shape = (batch_size, num_elements, 2, 2)
    local_matrix = diffusion_mid * dNdxdNdx * h[None, :, None, None]
    return local_matrix


def _compute_local_vector(
        source_fn: Callable, 
        element_nodes_coords: torch.Tensor,
        ):
    device: torch.device = element_nodes_coords.device
    float_type: torch.dtype = element_nodes_coords.dtype
    # Compute the length and midpoint of the elements
    # h.shape = midpoint.shape = (num_elements,)
    h = torch.abs(element_nodes_coords[:, 0] - element_nodes_coords[:, 1])
    midpoint = 0.5 * (element_nodes_coords[:, 0] + element_nodes_coords[:, 1])
    # Source function evaluation at the midpoint of each element
    # source_mid.shape = (num_elements, 1)
    source_mid = source_fn(midpoint)[:, None]
    # N_mid.shape = (2,)
    N_mid = torch.tensor([0.5, 0.5], dtype=float_type, device=device) 
    # local_vector.shape = (num_elements, 2)
    local_vector = source_mid * N_mid[None, :] * h[:, None]
    return local_vector

# ------------------- Assembly -----------------

def _assemble_mat_vec_prod(
        param: torch.Tensor,
        diffusion_param_fn,
        input_vec: torch.Tensor,
        nodes: torch.Tensor, 
        elements: torch.Tensor,
        ) -> torch.Tensor:
    # param.shape = (batch_size, dim(param))
    # input_vec.shape = output_vec.shape = (batch_size, num_dofs)
    device: torch.device = nodes.device
    float_type: torch.dtype = nodes.dtype
    batch_size = param.size(0)
    
    # Pre-compute global DOF mapping for better GPU utilization
    # global_dofs.shape = (num_elements, 2)
    global_dofs = elements
    # element_nodes_coords.shape = (num_elements, 2)
    element_nodes_coords = nodes[elements]
    
    # Gather input vectors with optimized indexing
    # input_vec_local.shape = (batch_size, num_elements, 2, 1)
    local_input_vec = input_vec[:, global_dofs].unsqueeze(-1)
    
    # Compute local matrices with vectorized operations
    local_mat = _compute_local_matrix(
            param=param, diffusion_param_fn=diffusion_param_fn, 
            element_nodes_coords=element_nodes_coords)
    
    # Local matrix-vector product with optimized einsum
    # local_mat_vec_prod.shape = (batch_size, num_elements, 2)
    local_mat_vec_prod = torch.einsum(
        "beik,bekj->beij", local_mat, local_input_vec).squeeze(-1)
    
    # Scatter with optimized memory access pattern
    src = local_mat_vec_prod.reshape(batch_size, -1)
    index = global_dofs.reshape(1, -1).expand(batch_size, -1)
    output_vec = torch.zeros_like(input_vec, dtype=float_type, device=device)
    output_vec.scatter_add_(dim=1, index=index, src=src)
    
    return output_vec


# --------------- Auxiliary functions for direct solving the linear system ------------
def _compute_local_matrix_single_element(
        param: torch.Tensor,
        diffusion_param_fn: Callable,
        element_nodes_coords: torch.Tensor,
        ):
    # param.shape = (batch_size, dim(param))
    assert param.ndim == 2
    # Compute the length and midpoint of the element: [nodes_coords[0], nodes_coords[1]]
    h: torch.Tensor = torch.abs(element_nodes_coords[0] - element_nodes_coords[1])
    midpoint: torch.Tensor = 0.5 * (element_nodes_coords[0] + element_nodes_coords[1])
    # Numerical integration over the current element using midpoint rule
    diffusion_mid = diffusion_param_fn(param, midpoint)[:, None, None]
    dNdx = torch.cat((-1.0 / h, 1.0 / h), dim=1)
    dNdxdNdx = torch.outer(dNdx, dNdx)
    local_matrix = diffusion_mid * dNdxdNdx * h
    # local_matrix.shape = (batch_size, 2, 2)
    return local_matrix

def _assemble(
        param: torch.Tensor,
        diffusion_param_fn: Callable,
        nodes: torch.Tensor, 
        elements: torch.Tensor,
        free_dofs: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    # param.shape = (dim(param))
    assert param.ndim == 1
    float_type = param.dtype
    # The number of nodes and number of degrees of freedom
    num_nodes: int = nodes.size(0)
    num_dofs: int =  num_nodes
    num_elems: int = elements.size(0)
    # Initialize arrays for Coordinate (triplet) format (I, J, V)
    # Each 1D element has a 2x2 matrix, so 4 non-zero entries.
    num_entries: int = 4 * num_elems
    I = torch.zeros(num_entries, dtype=torch.int64)
    J = torch.zeros(num_entries, dtype=torch.int64)
    V = torch.zeros(num_entries, dtype=float_type)
    # Assemble the global matrix and vector
    for idx, elem in enumerate(elements):
        # compute local matrix
        K_local = _compute_local_matrix_single_element(
            param=param[None, :], diffusion_param_fn=diffusion_param_fn, 
            element_nodes_coords=nodes[elem]).squeeze(0)
        # Define the mapping from local to global degrees of freedom (DoFs)
        global_dofs = torch.cat((elem, num_nodes + elem))
        global_rows, global_cols = torch.meshgrid(global_dofs, global_dofs)
        # Get the indices for the next 4 entries in our I, J, V arrays
        start_idx: int = 4 * idx
        end_idx: int = 4 * (idx + 1)
        # Store the row indices, column indices, and values
        I[start_idx:end_idx] = global_rows.flatten()
        J[start_idx:end_idx] = global_cols.flatten()
        V[start_idx:end_idx] = K_local.flatten()
    # Full global matrix
    indices = torch.stack([I, J])
    K_coo = torch.sparse_coo_tensor(indices=indices, values=V, 
                                    size=(num_dofs, num_dofs), 
                                    dtype=float_type)
    K_csr = K_coo.to_sparse_csr()
    # Remove rows and columns corresponding to Dirichlet degrees of freedom (DoFs)
    free_dof_mask = torch.zeros(num_dofs, dtype=torch.bool)
    free_dof_mask[free_dofs] = True
    # Create a mask for the triplets: keep an entry (I, J, V) if both I and J
    # are in the free_dofs set.
    triplet_mask = free_dof_mask[I] & free_dof_mask[J]
    # Filter the I, J, V triplets to keep only the active ones
    I_reduced = I[triplet_mask]
    J_reduced = J[triplet_mask]
    V_reduced = V[triplet_mask]
    # Remap the global indices to the local indices of the reduced matrix
    # First, create a mapping array
    num_free_dofs: int = len(free_dofs)
    dof_remapper = torch.zeros(num_dofs, dtype=torch.int64)
    dof_remapper[free_dofs] = torch.arange(num_free_dofs, dtype=torch.int64)
    # Then, apply the mapping
    I_mapped = dof_remapper[I_reduced]
    J_mapped = dof_remapper[J_reduced]
    # Convert to COO sparse matrix
    indices_reduced = torch.stack([I_mapped, J_mapped])
    K_reduced_coo = torch.sparse_coo_tensor(indices=indices_reduced, values=V_reduced, 
                                            size=(num_free_dofs, num_free_dofs), 
                                            dtype=float_type)
    # Convert to CSR sparse matrix
    K_reduced_csr = K_reduced_coo.to_sparse_csr()
    return K_reduced_csr, K_csr

def _sparse_solve_torch(
        mat_torch_csr: torch.Tensor, 
        vec_torch: torch.Tensor
        ) -> npt.NDArray:
        mat_scipy_csr = csr_matrix((mat_torch_csr.values().numpy(), 
                                  mat_torch_csr.col_indices().numpy(), 
                                  mat_torch_csr.crow_indices().numpy()))
        # Convert the load vector from torch.Tensor to np.array
        vec_numpy: npt.NDArray = vec_torch.numpy()
        # Solve the sparse linear system by scipy
        solution = spsolve(mat_scipy_csr, vec_numpy)
        return cast(npt.NDArray, solution)
