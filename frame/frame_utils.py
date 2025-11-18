from typing import List, Tuple, Callable, cast, Optional, Union, overload

import torch
from torch.nn import functional as F

from .mesh import Mesh1D

# ----------------------
# scaling 

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
        mesh_level: Mesh1D,
        ):
    device = mesh_level.device
    # Generate the mesh for this level
    num_nodes = mesh_level.num_nodes
    num_dofs = num_nodes
    dofs = torch.arange(num_dofs, dtype=torch.int64, device=device)
    free_dofs_mask = torch.ones(num_dofs, dtype=torch.bool, device=device)
    free_dofs_mask[dofs[mesh_level.dirichlet_mask]] = False
    free_dofs = dofs[free_dofs_mask]
    # Compute the diagonal matrix 
    laplacian_diag = _assemble_laplacian_diag(
        nodes=mesh_level.nodes, 
        elements=mesh_level.elements,
        free_dofs=free_dofs
    )
    # Compute the inverse of the diagonal matrix
    return laplacian_diag

def _compute_sqrt_laplacian_diag_inv_per_level(
        mesh_per_level: Tuple[Mesh1D, ...],
        ) -> torch.Tensor:
    num_levels = len(mesh_per_level)
    float_type = mesh_per_level[0].float_type
    device = mesh_per_level[0].device
    num_free_dofs_finest = 2 ** num_levels - 1
    sqrt_laplacian_diag_inv = torch.zeros(num_levels, num_free_dofs_finest, dtype=float_type, device=device)
    for idx_level in range(num_levels):
        num_free_dofs_current_level = 2 ** (idx_level + 1) - 1
        laplacian_diag = _assemble_laplacian_diag_level(mesh_level=mesh_per_level[idx_level])
        sqrt_laplacian_diag_inv[idx_level, :num_free_dofs_current_level] = torch.sqrt(1.0 / laplacian_diag)
    return sqrt_laplacian_diag_inv


# ----------------------

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

# -------------------



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

def _compute_quadrature_points_element_index_level(
        mesh_level: Mesh1D,
        quadrature_points: torch.Tensor,
        ):
    # Generate the mesh for this level
    nodes = mesh_level.nodes 
    elements = mesh_level.elements
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

def _diff_frame_level(
        mesh_level: Mesh1D,
        index_element_level: torch.Tensor,
        ) -> torch.Tensor:
    nodes = mesh_level.nodes 
    elements = mesh_level.elements
    free_dofs = mesh_level.free_dofs
    mat = _assemble_differential_matrix(nodes=nodes, elements=elements, free_dofs=free_dofs)
    return mat[index_element_level, :]

def _diff_frame_scaled_per_level(
        mesh_per_level: Tuple[Mesh1D, ...],
        index_element_per_level: torch.Tensor,
        sqrt_diag_inv_per_level: torch.Tensor,
        ) -> torch.Tensor:
    float_type = sqrt_diag_inv_per_level.dtype
    device = sqrt_diag_inv_per_level.device
    num_levels = len(mesh_per_level)
    num_coefficients = sum(2 ** (k + 1) - 1 for k in range(num_levels))
    num_points = index_element_per_level.size(0)
    scaled_mat = torch.zeros(num_points, num_coefficients, dtype=float_type, device=device)
    coeff_start_idx = 0
    for idx_level in range(num_levels):
        num_nodes_current_level = 2 ** (idx_level + 1) - 1
        coeff_end_idx = coeff_start_idx + num_nodes_current_level
        sqrt_diag_inv_current = sqrt_diag_inv_per_level[idx_level, :num_nodes_current_level]
        mat = _diff_frame_level(
            mesh_level=mesh_per_level[idx_level], 
            index_element_level=index_element_per_level[:, idx_level]
        )
        scaled_mat[:, coeff_start_idx:coeff_end_idx] = mat * sqrt_diag_inv_current
        coeff_start_idx = coeff_end_idx
    return scaled_mat

def _compute_quadrature_points_element_index_per_level(
        mesh_per_level: Tuple[Mesh1D, ...],
        quadrature_points: torch.Tensor,
        ) -> torch.Tensor:
    device = quadrature_points.device
    num_points = quadrature_points.size(0)
    num_levels = len(mesh_per_level)
    index_element_per_level = torch.zeros(num_points, num_levels, dtype=torch.int64, device=device)
    for idx_level in range(num_levels):
        index_element_per_level[:, idx_level] = _compute_quadrature_points_element_index_level(
            mesh_level=mesh_per_level[idx_level],
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

