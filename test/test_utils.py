import torch
import pytest
import pandas as pd
from torch.autograd import Variable

@pytest.fixture
def binary_sample_data():
    # Load and convert data
    with open('../data/samples.csv', 'r') as csv_file:
        data = pd.read_csv(csv_file)
    x, y = data[['x_1', 'x_2']].as_matrix(), data['y'].as_matrix()

    negative_class = torch.from_numpy(y) < 0
    y = torch.ones(y.shape).long()
    y[negative_class] = 0

    x = torch.from_numpy(x)

    x = Variable(x)
    y = Variable(y)

    return x, y


@pytest.fixture
def binary_categorical_sample_data():
    # Load and convert data
    with open('../data/samples.csv', 'r') as csv_file:
        data = pd.read_csv(csv_file)
    x, y = data[['x_1', 'x_2']].as_matrix(), data['y'].as_matrix()

    negative_class = torch.from_numpy(y) < 0
    y = torch.ones(y.shape).long()
    y[negative_class] = 0

    x = torch.from_numpy(x)
    l = torch.ones(x.size(0)).uniform_()
    x[:, 0] = 2
    x[:, 0][l < 0.67] = 1
    x[:, 0][l < 0.33] = 0
    x = Variable(x)
    y = Variable(y)

    return x, y
