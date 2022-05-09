# +
from autoforce.core.parameters.param_map import ParamMap, SymmetricParamMap


def test_1():
    dict_ = {1: 1.0, 2: 2.0}
    parmap = ParamMap(0.0, dict_)
    assert parmap.broadcast(((0, 1, 2),)) == (0.0, 1.0, 2.0)
    assert parmap.product((0, 1, 2), repeat=1) == {(0,): 0.0, (1,): 1.0, (2,): 2.0}


def test_2():
    dict_ = {1: 1.0, 2: 2.0}
    symparmap = SymmetricParamMap(0.0, dict_)
    assert symparmap.broadcast(((0, 1, 2),)) == (0.0, 1.0, 2.0)
    assert symparmap.product((0, 1, 2), repeat=1) == {(0,): 0.0, (1,): 1.0, (2,): 2.0}


def test_3():
    dict_ = {(2, 1): 1.0, (1, 3): 2.0}
    parmap = ParamMap(0.0, dict_)
    assert parmap.broadcast((1, (1, 2, 3))) == (0.0, 0.0, 2.0)
    assert parmap.product((1, 2), repeat=2) == {
        (1, 1): 0.0,
        (1, 2): 0.0,
        (2, 1): 1.0,
        (2, 2): 0.0,
    }


def test_4():
    dict_ = {(2, 1): 1.0, (1, 3): 2.0}
    symparmap = SymmetricParamMap(0.0, dict_)
    assert symparmap.broadcast((1, (1, 2, 3))) == (0.0, 1.0, 2.0)
    assert symparmap.product((1, 2), repeat=2) == {
        (1, 1): 0.0,
        (1, 2): 1.0,
        (2, 2): 0.0,
    }


if __name__ == "__main__":
    test_1()
    test_2()
    test_3()
    test_4()
