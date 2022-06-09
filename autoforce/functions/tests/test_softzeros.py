# +
import torch

from autoforce.core.functions import SoftZero_fn
from autoforce.functions.softzeros import Polynomial_sz, Sine_sz


def _test(f: SoftZero_fn) -> bool:
    f = Polynomial_sz(2)
    x = torch.tensor(0.0, requires_grad=True)
    y = f.function(x)
    assert y.isclose(x)
    (g,) = torch.autograd.grad(y, x)
    assert g.isclose(x)
    x = torch.tensor(1.0)
    y = f.function(x)
    assert y.isclose(x)
    x = torch.rand(10)
    y = f.function(x)
    assert (y > 0.0).all() and (y < 1.0).all()
    return True


def test_Polynomial_sz() -> bool:
    return _test(Polynomial_sz(2))


def test_Sine_sz() -> bool:
    return _test(Sine_sz())


if __name__ == "__main__":
    test_Polynomial_sz()
