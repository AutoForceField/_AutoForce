# +
import torch

import autoforce.cfg as cfg
from autoforce.core import ChemPar, ReducedPar
from autoforce.functions import Range_bi


def test_ChemPar() -> bool:
    def tensor(*args, **kwargs):
        return torch.tensor(*args, **kwargs, dtype=cfg.float_t)

    #
    s = ChemPar(values={1: 1.0})
    assert s[(1,)] == s[1] == 1.0 and s[2] is None
    s[:] = 3.0
    assert s[1] == s[2] == 3.0

    #
    s = ChemPar(values={1: 1.0}, default=3.0)
    assert s[1] == 1.0 and s[54345] == 3.0
    a = torch.tensor([1, 2])
    assert s(a).allclose(tensor([1.0, 3.0]))

    #
    s = ChemPar(values={(1, 2): 1.0})
    assert s[1, 2] == s[2, 1] == s[(1, 2)] == s[(2, 1)] == 1.0 and s[1, 3] is None
    s[:] = 2.0
    assert s[117, 75] == 2.0

    #
    s = ChemPar(values={(1, 2): 1.0}, permsym=False)
    assert s[1, 2] == 1.0 and s[2, 1] is None

    #
    s = ChemPar(values={(1, 2): 1.0}, default=3.0, permsym=False)
    assert s[1, 2] == 1.0 and s[2, 1] == 3.0

    #
    s = ChemPar(values={(1, 2): 1.0, (2, 1): 2.0}, default=3.0, permsym=False)
    assert s[1, 2] == 1.0 and s[2, 1] == 2.0

    #
    s = ChemPar(keylen=2)
    s[:] = 1.0
    s[1, 2] = 2.0
    assert s[1, 1] == s[2, 2] == 1.0 and s[1, 2] == s[2, 1] == 2.0

    #
    s = ChemPar(values={(1, 2): 1.0, (2, 2): 7.0}, default=3.0)
    a = torch.tensor(1)
    b = torch.tensor([1, 2, 3])
    assert (s(a, b) == tensor([3.0, 1.0, 3.0])).all()
    assert (s(b, b) == tensor([3.0, 7.0, 3.0])).all()

    s.as_dict([1])

    #
    const = Range_bi(0.0, 10.0)
    s = ChemPar(values={(1, 2): 1.0, (2, 2): 7.0}, default=3.0, bijection=const)
    assert s(a, b).allclose(tensor([3.0, 1.0, 3.0]))
    assert s(b, b).allclose(tensor([3.0, 7.0, 3.0]))

    return True


def test_ReducedPar():

    a = ChemPar(values={(1, 2): 1.0, (2, 2): 7.0}, default=3.0)
    b = ChemPar(values={(1, 2): 8.0, (2, 2): 2.0}, default=4.0)
    c = ReducedPar(a, b, op=max)
    d = c.as_dict([1, 2, 3], float)
    assert d[(1, 1)] == 4.0
    assert d[(1, 2)] == 8.0
    assert d[(2, 2)] == 7.0
    assert d[(2, 3)] == 4.0

    return True


if __name__ == "__main__":
    print(test_ChemPar())
    print(test_ReducedPar())
