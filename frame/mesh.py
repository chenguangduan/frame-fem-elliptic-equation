import abc

import torch 


class Mesh1D(abc.ABC):
    def __init__(
            self, 
            n: int,
            float_type: torch.dtype,
            device: torch.device,
            ) -> None:
        # Initialize device and float_type
        self.device: torch.device = device
        self.float_type: torch.dtype = float_type
        # Generate mesh
        self.num_nodes: int = n + 1
        self.nodes: torch.Tensor 
        self.elements: torch.Tensor
        self.nodes, self.elements = self._generate_nodes_elements()
        self.num_elements = self.elements.size(0)
        # Get Dirichlet degrees of freedom
        self.dirichlet_mask: torch.Tensor = self._compute_dirichlet_mask()
        self.free_dofs = self._compute_free_dofs()

    def to(self, device: torch.device):
        """ Move tensor atttibutes to device
        """
        device = torch.device(device)
        # Tensor attributes that need to be move
        tensor_attributes_name = [
            "nodes", "elements", "dirichlet_mask", "free_dofs"
        ]
        for attribute_name in tensor_attributes_name:
            # Get attributes 
            tensor = getattr(self, attribute_name)
            # Move tensor to device and then set the attributes
            setattr(self, attribute_name, tensor.to(device))
        # Set up the device attributes
        self.device = device 
        return self
    
    def set_float_type(self, float_type: torch.dtype):
        """ Set the precision of float tensor attributes
        """
        # Float tensor attributes that need to be set the float type
        self.nodes = self.nodes.to(float_type)
        # Set up the float_type attributes
        self.float_type = float_type
        return self
    
    def double(self):
        """ Set the precision of float tensor attributes to float64
        """
        self.nodes = self.nodes.double()
        self.float_type = torch.float64
        return self
    
    def float(self):
        """ Set the precision of float tensor attributes to float32
        """
        self.nodes = self.nodes.float()
        self.float_type = torch.float32
        return self
    
    def half(self):
        """ Set the precision of float tensor attributes to float16
        """
        self.nodes = self.nodes.half()
        self.float_type = torch.float16
        return self

    def _generate_nodes_elements(self):
        """ Generate nodes and elements of the mesh

        Returns:
            nodes (torch.Tensor): shape = (num_nodes,)
            elements (torch.Tensor): shape = (num_nodes - 1, 2)
        """
        nodes = torch.linspace(0.0, 1.0, self.num_nodes, dtype=self.float_type, device=self.device)
        nodes_indices = torch.arange(self.num_nodes, dtype=torch.int64, device=self.device).unsqueeze(-1)
        elements = torch.hstack((nodes_indices[:-1], nodes_indices[1:]))
        return nodes, elements
    
    def _compute_dirichlet_mask(self):
        """ Get the mask of nodes corresponding to Dirichlet boundary condition 

        Returns: 
            dirichlet_mask (torch.Tensor): shape = (num_nodes,)
        """
        eps: float = torch.finfo(self.float_type).eps
        # Get the mask of nodes at 0.0
        dirichlet_mask_left = torch.abs(self.nodes - 0.0) < eps
        # Get the mask of nodes at 1.0
        dirichlet_mask_right = torch.abs(self.nodes - 1.0) < eps
        dirichlet_mask = dirichlet_mask_left | dirichlet_mask_right 
        return dirichlet_mask.to(self.device)
    
    def _compute_free_dofs(self):
        num_dofs = self.num_nodes
        dofs = torch.arange(num_dofs, dtype=torch.int64, device=self.device)
        dirichlet_dofs = dofs[self.dirichlet_mask]
        free_dofs_mask = torch.ones(num_dofs, dtype=torch.bool, device=self.device)
        free_dofs_mask[dirichlet_dofs] = False
        return dofs[free_dofs_mask]
    
