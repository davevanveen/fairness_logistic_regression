import torch
import pytest
import h5py
import pandas as pd
from torch.autograd import Variable


def split_data(x, y, pct):
    nsamples = len(y)
    nsamples_val = int(pct * nsamples)
    srs = torch.utils.data.sampler.SubsetRandomSampler(range(nsamples))
    idxs = []
    for n, i in enumerate(srs):
        idxs.append(i)
        if n == nsamples_val - 1:
            x_val = x[idxs]
            y_val = y[idxs]
            idxs = []
    x = x[idxs]
    y = y[idxs]

    return x, y, x_val, y_val


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
    cat = torch.ones(x.size(0)).uniform_()
    x[:, 0] = 2
    x[:, 0][cat < 0.67] = 1
    x[:, 0][cat < 0.33] = 0
    x = Variable(x)
    y = Variable(y)

    return x, y


@pytest.fixture
def adult_sample_data():
    # Load and convert data
    f = h5py.File('../data/data_31.h5', 'r')
    data = Variable(torch.from_numpy(f['data'].value))
    target = Variable(torch.from_numpy(f['target'].value))
    f.close()

    x, y, _, _1 = split_data(data, target, 0.25)  # Make it smaller
    return x, y
