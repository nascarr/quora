import pytest
from ensemble import *

def test_find_k_dirs():
    k_dirs = find_k_dirs('models/Jan_19_2019__12_10_14', 5)
    assert k_dirs == ['models/Jan_19_2019__12_10_14', 'models/Jan_19_2019__12_32_38',
                      'models/Jan_19_2019__12_55_05', 'models/Jan_19_2019__13_17_28', 'models/Jan_19_2019__13_39_54']
