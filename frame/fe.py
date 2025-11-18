import abc
from typing import Tuple, Union, overload

import numpy.typing as npt

import torch

from .pde import ParametricEllipticPDE
from .mesh import Mesh1D

from .fe_utils import (
    _assemble_load,
    _assemble_stiffness_vec_prod,
    _assemble,
    _sparse_solve_torch
)

from .frame_utils import (
    _compute_sqrt_laplacian_diag_inv_per_level,
    _compute_quadrature_points_element_index_per_level,
    _diff_frame_scaled_per_level,
    _frame_synthesis_fn,
    _frame_analysis_fn,
    _assemble_parametric_diag,
    _diff_frame_scaled_vec_prod_per_level,
    _diff_frame_scaled_trans_vec_prod_per_level
)


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
        self.num_levels: int = num_levels
        self.mesh_per_level: Tuple[Mesh1D, ...] = tuple(
            Mesh1D(n=2 ** (idx_level + 1), float_type=self.float_type, device=self.device)
            for idx_level in range(self.num_levels))
        self.nodes: torch.Tensor = self.mesh_per_level[-1].nodes
        self.num_nodes = self.nodes.size(0)
        self.elements: torch.Tensor = self.mesh_per_level[-1].elements
        self.num_dofs: int = self.num_nodes
        self.num_elements: int = self.elements.size(0)
        self.free_dofs: torch.Tensor = self.mesh_per_level[-1].free_dofs
        self.num_free_dofs: int = self.free_dofs.size(0)
        # Combination of differential operator and frame
        self.num_coefficients = sum(2 ** (k + 1) - 1 for k in range(num_levels))
        self.sqrt_diag_inv_per_level = _compute_sqrt_laplacian_diag_inv_per_level(
            mesh_per_level=self.mesh_per_level
        )
        # quadrature points and their element index
        element_nodes_coords = self.nodes[self.elements]
        quadrature_points = 0.5 * (element_nodes_coords[:, 0] + element_nodes_coords[:, 1])
        self.index_elements_per_level = _compute_quadrature_points_element_index_per_level(
            mesh_per_level=self.mesh_per_level,
            quadrature_points=quadrature_points
        )
        # explicit differential-frame matrix
        self.diff_frame_scaled_per_level = _diff_frame_scaled_per_level(
            mesh_per_level=self.mesh_per_level,
            index_element_per_level=self.index_elements_per_level,
            sqrt_diag_inv_per_level=self.sqrt_diag_inv_per_level
        )

    @overload 
    def to(self, input: torch.device):
        ...

    @overload
    def to(self, input: torch.dtype):
        ...

    def to(self, input: Union[torch.device, torch.dtype]):
        if isinstance(input, torch.device):
            return self.set_device(input)
        elif isinstance(input, torch.dtype):
            return self.set_float_precision(input)  
        else:
            raise TypeError("Unsupported data type")

    def set_device(self, device: torch.device):
        device = torch.device(device)
        # List of tensor attributes to be moved
        tensor_attributes_name = [
            "nodes", "elements", "dofs", "dirichlet_dofs", 
            "free_dofs_mask", "free_dofs", "sqrt_diag_inv_per_level",
            "index_elements_per_level", "diff_frame_scaled_per_level"
        ]
        for attribute_name in tensor_attributes_name:
            tensor = getattr(self, attribute_name)
            setattr(self, attribute_name, tensor.to(device))
        for mesh in self.mesh_per_level:
            mesh = mesh.to(device)
        self.device = device 
        return self
    
    def cpu(self):
        return self.to(torch.device("cpu"))
    
    def set_float_precision(self, float_type: torch.dtype):
        if float_type == torch.float16:
            return self.half()
        elif float_type == torch.float32:
            return self.float()
        elif float_type == torch.float64:
            return self.double()
        else:
            raise ValueError(f"Invalid float type {float_type}.")
    
    def double(self):
        # List of tensor attributes
        float_tensor_attributes_name = [
            "nodes", "sqrt_diag_inv_per_level", "diff_frame_scaled_per_level"
        ]
        for attribute_name in float_tensor_attributes_name:
            float_tensor = getattr(self, attribute_name)
            setattr(self, attribute_name, float_tensor.double())
        for mesh in self.mesh_per_level:
            mesh = mesh.double()
        self.float_type = torch.float64 
        return self
    
    def float(self):
        # List of tensor attributes
        float_tensor_attributes_name = [
            "nodes", "sqrt_diag_inv_per_level", "diff_frame_scaled_per_level"
        ]
        for attribute_name in float_tensor_attributes_name:
            float_tensor = getattr(self, attribute_name)
            setattr(self, attribute_name, float_tensor.float())
        for mesh in self.mesh_per_level:
            mesh = mesh.float()
        self.float_type = torch.float32
        return self
    
    def half(self):
        # List of tensor attributes
        float_tensor_attributes_name = [
            "nodes", "sqrt_diag_inv_per_level", "diff_frame_scaled_per_level"
        ]
        for attribute_name in float_tensor_attributes_name:
            float_tensor = getattr(self, attribute_name)
            setattr(self, attribute_name, float_tensor.half())
        for mesh in self.mesh_per_level:
            mesh = mesh.half()
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
    
    def assemble_frame_stiffness_coef_prod_decomposition(
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
    
    def assemble_frame_stiffness_coef_prod_decomposition_mat(
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
    
    def assemble_frame_stiffness_coef_prod(
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
        mat_vec_prod = _assemble_stiffness_vec_prod(
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

