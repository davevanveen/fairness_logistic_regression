import sys
sys.path.append('../fair_regression')

from FairLogReg import FairLogisticRegression
from test_utils import *  # noqa


def test_plain_logreg(binary_sample_data):
    x, y = binary_sample_data

    lr = FairLogisticRegression()
    lr.fit(x, y, 0)

    print(lr.score(x, y))


def test_plain_validated_logreg(binary_sample_data):
    x, y = binary_sample_data

    lr = FairLogisticRegression(validate=0.5, print_freq=4)
    lr.fit(x, y, 0)

    print(lr.score(x, y))


# def test_fair_logreg(adult_sample_data):
#     x, y = adult_sample_data

#     lr = FairLogisticRegression(l_fair=0.1, validate=0.1, print_freq=4)
#     lr.fit(x, y, 29)  # 29 -> 'sex_Female'

#     print(lr.score(x, y))


def test_load_save_logreg(binary_sample_data):
    x, y = binary_sample_data

    lr = FairLogisticRegression(validate=0.5, print_freq=4, ftol=1e-2)
    lr.fit(x, y, 0)

    fn = 'test_save.pt'
    lr.save(fn)

    lr2 = FairLogisticRegression.load(fn)
    lr2.fit(x, y, 0)

    print(lr2.score(x, y))


def test_real_data(female_adult_data):
    s, x_train, y_train, x_test, y_test = female_adult_data

    lr = FairLogisticRegression(validate=0.4, print_freq=4, l_fair=0.1)
    lr.fit(x_train, y_train, s)

    print(lr.score(x_test, y_test))
