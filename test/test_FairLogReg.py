import sys
sys.path.append('../fair_regression')

from FairLogReg import FairLogisticRegression
from test_utils import *  # noqa


# def test_plain_logreg(binary_sample_data):
#     x, y = binary_sample_data

#     lr = FairLogisticRegression()
#     lr.fit(x, y, 0)

#     print(lr.score(x, y))


# def test_plain_validated_logreg(binary_sample_data):
#     x, y = binary_sample_data

#     lr = FairLogisticRegression(validate=0.5, print_freq=4)
#     lr.fit(x, y, 0)

#     print(lr.score(x, y))


# def test_fair_logreg(adult_sample_data):
#     x, y = adult_sample_data

#     lr = FairLogisticRegression(l_fair=0.1, validate=0.1, print_freq=4)
#     lr.fit(x, y, 29)  # 29 -> 'sex_Female'

#     print(lr.score(x, y))


# def test_load_save_logreg(binary_sample_data):
#     x, y = binary_sample_data

#     lr = FairLogisticRegression(validate=0.5, print_freq=4, ftol=1e-2)
#     lr.fit(x, y, 0)

#     fn = 'test_save.pt'
#     lr.save(fn)

#     lr2 = FairLogisticRegression.load(fn)
#     lr2.fit(x, y, 0)

#     print(lr2.score(x, y))


def test_real_data(female_adult_data):
    s, x_train, y_train = female_adult_data
    # This data is huge, so just take a small part to test
    _, _1, x_train, y_train = split_data(x_train, y_train, 0.1)
    x_train, y_train, x_test, y_test = split_data(x_train, y_train, 0.3)

    lr = FairLogisticRegression(validate=0.4, print_freq=4, l_fair=1000, batch_fairness=True)
    lr.fit(x_train, y_train, s)

    pure_lr = FairLogisticRegression(validate=0.4, print_freq=4, l_fair=0.0)
    pure_lr.fit(x_train, y_train, s)

    print('With fairness: {}'.format(lr.score(x_test, y_test)))
    print('Without fairness: {}'.format(pure_lr.score(x_test, y_test)))


def test_joint(joint_adult_data):
    s, x_train, y_train = joint_adult_data
    # This data is huge, so just take a small part to test
    x_train, y_train, x_test, y_test = split_data(x_train, y_train, 0.3)

    lr = FairLogisticRegression(validate=0.4, print_freq=4, l_fair=10, batch_fairness=True)
    lr.fit(x_train, y_train, s)

    print('With fairness: {}'.format(lr.score(x_test, y_test)))


def test_pairwise(pairwise_adult_data):
    s, x_train, y_train = pairwise_adult_data
    # This data is huge, so just take a small part to test
    x_train, y_train, x_test, y_test = split_data(x_train, y_train, 0.3)

    lr = FairLogisticRegression(validate=0.4, print_freq=4, l_fair=10, batch_fairness=True)
    lr.fit(x_train, y_train, s)

    print('With fairness: {}'.format(lr.score(x_test, y_test)))
