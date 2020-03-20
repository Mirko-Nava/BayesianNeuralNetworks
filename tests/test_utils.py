import pytest
from bnn import utils
from collections.abc import Iterable


def test_item_or_list(get_item_or_list):
    for example in get_item_or_list:
        e = utils._item_or_list(example)

        if len(example) == 1:
            assert e == example[0]
        else:
            assert e == example


def test_single(get_single):
    for example in get_single:
        e = utils._single(example)
        assert isinstance(e, tuple)
        assert len(e) == 1

        if not isinstance(example, Iterable):
            assert len(set(e)) == 1


def test_pair(get_pair):
    for example in get_pair:
        e = utils._pair(example)
        assert isinstance(e, tuple)
        assert len(e) == 2

        if not isinstance(example, Iterable):
            assert len(set(e)) == 1


def test_triple(get_triple):
    for example in get_triple:
        e = utils._triple(example)
        assert isinstance(e, tuple)
        assert len(e) == 3

        if not isinstance(example, Iterable):
            assert len(set(e)) == 1


def test_apply_wb(get_apply_wb):
    for example in get_apply_wb:
        m, fn, r = example
        results = utils.apply_wb(m, fn)

        if r is None:
            assert results is None
        else:
            assert results == r


def test_traverse(get_traverse):
    for example in get_traverse:
        m, fn, r = example
        results = utils.traverse(m, fn)

        if r is None:
            assert results is None
        else:
            assert results == r
