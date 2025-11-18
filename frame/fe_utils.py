import abc
from typing import List, Tuple, Callable, cast, Optional, Union, overload

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

import torch
from torch.nn import functional as F

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
    local_vector = _compute_local_load(source_fn=source_fn, element_nodes_coords=nodes[elements])
    # global_dofs.shape = (num_elements, 2)
    global_dofs = elements
    # scatter
    src = local_vector.reshape(-1)
    index = global_dofs.reshape(-1)
    F_global.scatter_add_(dim=0, index=index, src=src)
    return F_global

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

def _compute_local_load(
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

def _assemble_stiffness_vec_prod(
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

