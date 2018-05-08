import sys
sys.path.append('../fair_regression')

from FairLogReg import FairLogisticRegression
from test_utils import *  # noqa


def test_load_save(joint_adult_data):
    s, x, y = joint_adult_data

    lr = FairLogisticRegression(validate=0.5, print_freq=4, ftol=1e-2)
    lr.fit(x, y, 0)

    fn = 'test_save.pt'
    lr.save(fn)

    lr2 = FairLogisticRegression.load(fn)
    lr2.fit(x, y, 0)

    print(lr2.score(x, y))


def test_no_reg(female_adult_data):
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


# Driver for testing all of the combinations of dataset types and penalty types
def run_test(dataset, pt):
    s, x_train, y_train = dataset
    # This data is huge, so just take a small part to test
    x_train, y_train, x_test, y_test = split_data(x_train, y_train, 0.3)

    lr = FairLogisticRegression(validate=0.4, print_freq=4, l_fair=10, batch_fairness=True, penalty_type=pt)
    lr.fit(x_train, y_train, s)

    print('With fairness: {}'.format(lr.score(x_test, y_test)))


@pytest.mark.parametrize('pt', ['group', 'individual', 'novel'])
def test_univariate_penalties(female_adult_data, pt):
    run_test(female_adult_data, pt)


@pytest.mark.parametrize('pt', ['group', 'individual', 'novel'])
def test_joint_penalties(joint_adult_data, pt):
    run_test(joint_adult_data, pt)


@pytest.mark.parametrize('pt', ['group', 'individual', 'novel'])
def test_pairwise_penalties(pairwise_adult_data, pt):
    run_test(pairwise_adult_data, pt)
