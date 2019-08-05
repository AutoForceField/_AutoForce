#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch.nn import Module
from theforce.util.util import iterable


class ParametricPotential(Module):

    def __init__(self):
        super().__init__()
        self.params = []

    def forward(self, atoms_or_loc, forces=False, enable_grad=True):
        with torch.set_grad_enabled(enable_grad):
            if forces:
                f = 0
            e = 0
            for loc in iterable(atoms_or_loc):
                _e = self.calculate(loc, forces=forces)
                if forces:
                    _e, _f = _e
                    f = f + _f
                e = e + _e
            if forces:
                return e, f
            else:
                return e

    @property
    def state(self):
        return self.__class__.__name__+'({})'.format(self.state_args)


class PairPot(ParametricPotential):

    def __init__(self, a, b, radial):
        super().__init__()
        self.a = a
        self.b = b
        self.radial = radial
        try:
            self.params += radial.params
        except AttributeError:
            pass

    def calculate(self, loc, forces=False):
        loc.select(self.a, self.b, bothways=False)
        d = loc.r.pow(2).sum(dim=-1, keepdim=True).sqrt()
        e = self.radial(d, grad=forces)
        if forces:
            e, g = e
            f = -g*loc.r/d
            f = torch.zeros(loc.natoms, 3).index_add(
                0, loc.j, f).index_add(0, loc.i, -f)
            return e.sum(), f
        else:
            return e.sum()

    @property
    def state_args(self):
        return '{}, {}, {}'.format(self.a, self.b, self.radial.state)


def test():
    from theforce.descriptor.atoms import TorchAtoms
    from theforce.math.radial import RepulsiveCore
    torch.set_default_tensor_type(torch.DoubleTensor)

    V = PairPot(55, 55, RepulsiveCore())
    a = TorchAtoms(positions=[(0, 0, 0), (2, 0, 0), (0, 2, 0)],
                   numbers=[55, 55, 55], cell=[10, 10, 10], pbc=False)
    a.update(cutoff=5., posgrad=True)
    e, f = V(a, forces=True)
    e.backward()
    print(a.xyz.grad.allclose(-f))
    print(V.state)


if __name__ == '__main__':
    test()

