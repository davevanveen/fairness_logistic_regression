import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

def to_Variables(*args, cuda=False):
    ret = []
    for arg in args:
        if cuda:
            ret.append(Variable(arg).cuda())
        else:
            ret.append(Variable(arg))

    return ret


def check_and_convert_tensor(x):
    if torch.is_tensor(x):
        return x.cpu().numpy()
    else:
        return x.data.cpu().numpy()


n_features = 11
n_samples = 250
n_classes = 2

# Should give us a random label vector with entries 1 and -1
test_labels = torch.ones(n_samples).long()
test_labels[torch.ones(test_labels.shape).uniform_() > 0.5] = -1

# Generate random test data
test_data = torch.rand(n_samples, n_features).double()

# s indices
s = [8]

# Convert the 8th column of x to be categorical
l = torch.ones(test_data[:, s[0]].shape).uniform_()
test_data[:, s[0]] = 2
test_data[:, s[0]][l < 0.67] = 1
test_data[:, s[0]][l < 0.33] = 0

# Conversions to variables for use inside of everything
x = Variable(test_data).double()
y = Variable(test_labels).long()

# Generate random weights
w = Variable(torch.rand(n_features, 1)).double()




# Convert the data into a torch dataset
ds = TensorDataset(test_data, test_labels)
dl = DataLoader(ds, batch_size=32, shuffle=True)

# Enumerate through the data
for i, data in enumerate(dl):
    inputs, labels = to_Variables(*data)
    dtype = type(inputs.data)

    # Build up the indices
    fairness_idxs = []
    vals = []
    divisors = []
    for col in s:
        vals.append(np.unique(check_and_convert_tensor(test_data)[:, col]))
        fairness_idxs.append([])
        prod = 1
        for val in vals[-1]:
            idxs = (test_data[:, col] == float(val)).nonzero().squeeze()
            prod *= len(idxs)
            fairness_idxs[-1].append(idxs)
        divisors.append(prod)

    # Actually calculate the penalty
    penalty = 0
    for idxs, val, div in zip(fairness_idxs, vals, divisors):
        # Should be size (n_classes test_data self.minibatch_size)
        local_dots = torch.mm(inputs, w)
        for idx in idxs:
            # Because of broadcasting rules, this should create a matrix of absolute differences
            # of size (self.minibatch_size test_data |S_i: class[idx]|)
            diff = torch.abs(labels.view(1, -1) - y[idx].view(-1, 1)).type(dtype)

            # Size: (n_classes test_data |S_i: class[idx]|)
            non_local_dot = torch.mm(x[idx], w)

            cross_term = 0
            for local_dot in local_dots.t():
                # Individual version
                cross_term = cross_term + (local_dot - non_local_dot) ** 2
                penalty = penalty + ((diff / div) * cross_term).sum()

                # # Group version
                # cross_term = cross_term + (local_dot - non_local_dot)
                # penalty = penalty + ((diff / div) * cross_term).sum() ** 2

    print(penalty)