# +
import torch

from autoforce.functions import Range_bi


def test_Range_bi() -> bool:
    r = Range_bi(0.0, 1.0)
    x = torch.tensor([0.0, 0.5, 1.0])
    test = r.function(r.inverse(x)).allclose(x)
    return test


if __name__ == "__main__":
    a = test_Range_bi()
    print(a)
