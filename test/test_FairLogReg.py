import sys
sys.path.append('../fair_regression')

import pytest
from FairLogReg import FairLogisticRegression
from test_utils import *

def test_plain_logreg(binary_sample_data):
    x, y = binary_sample_data

    lr = FairLogisticRegression()
    lr.fit(x, y, 0)

    print(lr.score(x, y))


def test_fair_logreg(binary_categorical_sample_data):
    x, y = binary_categorical_sample_data

    lr = FairLogisticRegression(l_fair=0.1)
    lr.fit(x, y, 0)

    print(lr.score(x, y))