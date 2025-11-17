import abc
from typing import Callable


class ParametricEllipticPDE(abc.ABC):
    def __init__(
            self, 
            diffusion_param_fn: Callable, 
            source_fn: Callable
            ) -> None:
        self.diffusion_param_fn: Callable = diffusion_param_fn
        self.source_fn: Callable = source_fn

    def get_source_fn(self) -> Callable:
        return self.source_fn
    
    def get_diffusion_param_fn(self) -> Callable:
        return self.diffusion_param_fn
    

