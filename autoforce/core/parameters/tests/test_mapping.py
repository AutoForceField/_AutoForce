# +
from __future__ import annotations

from autoforce.core.parameters import AsymmetricMapping, SymmetricMapping


def test_1():
    dict_ = {1: 1.0, 2: 2.0}
    param = AsymmetricMapping(0.0, dict_)
    assert param[0] == param[(0,)] == 0.0
    assert param[1] == param[(1,)] == 1.0
    assert param.broadcast(((0, 1, 2),)) == (0.0, 1.0, 2.0)
    assert param.product((0, 1, 2), repeat=1) == {(0,): 0.0, (1,): 1.0, (2,): 2.0}


def test_2():
    dict_ = {1: 1.0, 2: 2.0}
    param = SymmetricMapping(0.0, dict_)
    assert param[0] == param[(0,)] == 0.0
    assert param[1] == param[(1,)] == 1.0
    assert param.broadcast(((0, 1, 2),)) == (0.0, 1.0, 2.0)
    assert param.product((0, 1, 2), repeat=1) == {(0,): 0.0, (1,): 1.0, (2,): 2.0}


def test_3():
    dict_ = {(2, 1): 1.0, (1, 3): 2.0}
    param = AsymmetricMapping(0.0, dict_)
    assert param[(2, 1)] == param[2, 1] == 1.0
    assert param[(1, 2)] == param[1, 2] == 0.0
    assert param.broadcast((1, (1, 2, 3))) == (0.0, 0.0, 2.0)
    assert param.product((1, 2), repeat=2) == {
        (1, 1): 0.0,
        (1, 2): 0.0,
        (2, 1): 1.0,
        (2, 2): 0.0,
    }


def test_4():
    dict_ = {(2, 1): 1.0, (1, 3): 2.0}
    param = SymmetricMapping(0.0, dict_)
    assert param[(2, 1)] == param[(1, 2)] == 1.0
    assert param.broadcast((1, (1, 2, 3))) == (0.0, 1.0, 2.0)
    assert param.broadcast(((1, 2, 3), 1)) == (0.0, 1.0, 2.0)
    assert param.product((1, 2), repeat=2) == {
        (1, 1): 0.0,
        (1, 2): 1.0,
        (2, 2): 0.0,
    }


if __name__ == "__main__":
    test_1()
    test_2()
    test_3()
    test_4()
