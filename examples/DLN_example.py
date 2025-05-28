import torch
import torch.nn as nn
from typing import List, Dict, Any, Union

torch.serialization.add_safe_globals([__import__("numpy").core.multiarray.scalar])

class DLN(nn.Module):
    @classmethod
    def make_rectangular(
        cls,
        input_dim: int,
        output_dim: int,
        L: int,
        w: int,
        gamma: float,
    ) -> "DLN":
        """
        Factory for a width-w rectangular DLN with weight variance w^{-γ}.
        """
        init_var = w ** (-gamma)
        instance = cls([input_dim] + [w] * (L - 1) + [output_dim], init_var)

        instance.rectangular = True
        instance.width = w
        instance.gamma = gamma
        return instance

    @classmethod
    def from_matrix(cls, A: torch.Tensor) -> "DLN":
        """
        Convert a single linear map (matrix) into a depth-1 DLN.
        """
        out_dim, in_dim = A.shape
        net = cls([in_dim, out_dim])
        net.linears[0].weight.data.copy_(A)
        return net

    def __init__(self, dims: List[int], init_variance: float = 1.0) -> None:
        super().__init__()
        self.dims: List[int] = list(dims)
        self.input_dim: int = dims[0]
        self.output_dim: int = dims[-1]
        self.hidden_dims: List[int] = dims[1:-1]
        self.hidden_size: int | None = self.hidden_dims[0] if self.hidden_dims else None
        self.L: int = len(dims) - 1
        self.init_variance: float = float(init_variance)

        self.rectangular: bool = len(set(self.hidden_dims)) == 1 if self.hidden_dims else False
        self.width: int | None = self.hidden_dims[0] if self.rectangular else None
        self.gamma: float | None = None  

        self.linears = nn.ModuleList(
            nn.Linear(d_in, d_out, bias=False) for d_in, d_out in zip(dims[:-1], dims[1:])
        )
        for layer in self.linears:
            layer.weight.data.normal_(0.0, self.init_variance)

    def get_config(self) -> Dict[str, Any]:
        """
        Return a json-serialisable dict that completely specifies the network.
        """
        return {
            "dims": self.dims,
            "init_variance": self.init_variance,
            "rectangular": self.rectangular,
            "width": self.width,
            "gamma": self.gamma,
        }

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "DLN":
        """
        Re-instantiate a DLN from get_config() output.
        """
        if cfg.get("rectangular"):
            return cls.make_rectangular(
                cfg["dims"][0],
                cfg["dims"][-1],
                len(cfg["dims"]) - 1,
                cfg["width"],
                cfg["gamma"],
            )
        return cls(cfg["dims"], cfg["init_variance"])