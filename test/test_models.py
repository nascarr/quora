import pytest
from models import *


@pytest.mark.parametrize("test_input, expected", [
    ([12, 12, 11, 10], [2, 1, 1]),
    ([10, 9, 8, 7, 6], [1, 1, 1, 1, 1]),
    ([11, 11, 11], [3]),
])
def test_section_sizes(test_input, expected):
    test_input = torch.tensor(test_input)
    expected = torch.tensor(expected)
    assert torch.equal(section_sizes(test_input), expected)


@pytest.mark.parametrize("test_input, expected", [
    (([12, 12, 11, 10], [4]), [12]),
    (([0.0, 1, 2.2, -3.5], [4]), [2.2]),
    (([[-1, -3, -5][-2, -4, 0]], [1, 1]), [-1]),
])
def test_max_packed(test_input, expected):
    test_input = [torch.tensor(t) for t in test_input]
    expected = torch.tensor(expected)
    assert max_packed(*test_input) == expected
