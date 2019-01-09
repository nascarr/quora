from unittest import TestCase

import pytest
from models import *
import numpy as np

def arr_to_tensors(arrs):
    tensors = [torch.tensor(a) for a in arrs]
    return tensors


def array_1():
    result = np.array(
    [[[-1, -1, -1, -1],
      [-2, -2, -2, -2]],
     [[-2, -2, -2, -2],
      [-4, -4, -4, -4]],
     [[3, 3, 3, 3],
      [0, 0, 0, 0]]], dtype=float)
    return result


def out_array_1():
    result = np.array(
    [[3, 3, 3, 3],
     [-4, -4, -4, -4]], dtype=float)
    return result


def max_array_1():
    result = np.array(
    [[3, 3, 3, 3],
     [-2, -2, -2, -2]], dtype=float)
    return result

@pytest.fixture
def mean_array_1():
    result = np.array(
    [[0, 0, 0, 0],
     [-3, -3, -3, -3]], dtype=float)
    return result


@pytest.mark.parametrize("test_input, expected_sizes, expected_section_lengths", [
    ([12, 12, 11, 10], [2, 1, 1], [12, 11, 10]),
    ([10, 9, 8, 7, 6], [1, 1, 1, 1, 1], [10, 9, 8, 7, 6]),
    ([11, 11, 11], [3], [11]),
])
def test_section_sizes_and_lengths(test_input, expected_sizes, expected_section_lengths):
    test_input,  expected_sizes, expected_section_lengths = arr_to_tensors([test_input,  expected_sizes, expected_section_lengths])
    assert torch.equal(section_sizes_and_lengths(test_input)[0], expected_sizes)
    assert torch.equal(section_sizes_and_lengths(test_input)[1], expected_section_lengths)


@pytest.mark.parametrize("test_input, test_lengths, expected", [
    (np.zeros((3,2,5)),
     [3, 2], np.zeros((2,5))),
    (array_1(), [3, 2], max_array_1())
])
def test_max_packed(test_input, test_lengths, expected):
    test_input, test_lengths, expected = arr_to_tensors([test_input, test_lengths, expected])
    assert torch.equal(max_packed(test_input, test_lengths), expected)


@pytest.mark.parametrize("test_input, test_lengths, expected", [
    (np.zeros((3, 2, 5)),
     [3, 2], np.zeros((2, 5))),
    (array_1(), [3, 2], mean_array_1())
])
def test_mean_packed(test_input, test_lengths, expected):
    test_input, test_lengths, expected = arr_to_tensors([test_input, test_lengths, expected])
    assert torch.equal(mean_packed(test_input, test_lengths), expected)

@pytest.mark.parametrize("test_input, test_lengths, expected_out, expected_max, expected_mean", [
    (array_1(),
    [3, 2],
    out_array_1(),
    max_array_1(),
    mean_array_1())
])
def test_out_max_mean_packed(test_input, test_lengths, expected_out, expected_mean, expected_max):
    test_input, test_lengths, expected_out, expected_mean, expected_max = arr_to_tensors(
        [test_input, test_lengths, expected_out, expected_mean, expected_max])
    assert torch.equal(out_max_mean_packed(test_input, test_lengths)[0], expected_out)
    assert torch.equal(out_max_mean_packed(test_input, test_lengths)[1], expected_max)
    assert torch.equal(out_max_mean_packed(test_input, test_lengths)[2], expected_mean)