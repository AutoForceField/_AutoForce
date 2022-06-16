# +
import torch
from torch import Tensor

from autoforce.core.modules import Regression


class LeastSquares(Regression):
    def fit(self, design_matrix: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        return torch.linalg.lstsq(design_matrix, targets).solution

    def is_deterministic(self) -> bool:
        return False
