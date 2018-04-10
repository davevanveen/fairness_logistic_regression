import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
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


class FairLogisticRegression():
    def __init__(self, lr=0.01, n_classes=None, ftol=1e-9, tolerance_grad=1e-5,
                 fit_intercept=True, n_epochs=32, l_fair=0.0, l2=0.0, l1=0.0,
                 minibatch_size=32, n_jobs=1):
        self.lr = lr
        self.n_classes = n_classes
        self.ftol = ftol
        self.tolerance_grad = tolerance_grad
        self.fit_intercept = fit_intercept
        self.n_epochs = n_epochs
        self.minibatch_size = minibatch_size
        self.n_jobs = n_jobs

        self.l_fair = l_fair
        self.l2 = l2
        self.l1 = l1

        self.predict_fn = torch.nn.LogSoftmax(dim=1)

        self.model = None
        self.optimizer = None
        self.loss = None
        self.n_features = None
        self.n_samples = None
        # self.y_diff = None
        self.fairness_idxs = None
        self.vals = None
        self.divisors = None

        self.training_errors_ = []
        self.validation_errors_ = []

    def build_model(self):
        # We don't need the softmax layer here since CrossEntropyLoss already
        # uses it internally.
        model = torch.nn.Linear(int(self.n_features), int(self.n_classes), bias=self.fit_intercept)
        return model

    def fit(self, x, y, s):
        # Make sure that s is a list for use in the code below
        if not isinstance(s, list):
            s = [s]

        # Convert data into a tensor dataset so that we can easily shuffle and mini-batch
        ds = TensorDataset(x.data, y.data)
        loader = DataLoader(ds, batch_size=self.minibatch_size, shuffle=True, num_workers=self.n_jobs)

        if self.n_samples is None or self.n_features is None:
            self.n_samples, self.n_features = x.size()

        if self.n_classes is None:
            self.n_classes = y.data.max() + 1

        if self.model is None:
            self.model = self.build_model().type(type(x.data))

        if self.loss is None:
            self.loss = torch.nn.CrossEntropyLoss(size_average=True).type(type(x.data))

        if self.optimizer is None:
            # self.optimizer = optim.LBFGS(self.model.parameters(), lr=self.lr, tolerance_grad=self.tolerance_grad,
            #                              tolerance_change=self.ftol)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2)

        for epoch in range(self.n_epochs):
            current_loss = 0
            for i, data in enumerate(loader):
                inputs, labels = to_Variables(*data, cuda=torch.cuda.is_available())

                def closure():
                    self.optimizer.zero_grad()

                    fx = self.model.forward(inputs)
                    output = self.loss.forward(fx, labels)
                    if self.l1:
                        output += self.l1_penalty()
                    if self.l_fair:
                        output += self.fairness_penalty(inputs, labels, x, y, s)
                    output.backward(retain_graph=True)
                    return output

                self.optimizer.step(closure)

        return self

    def predict(self, x):
        if self.model is None:
            raise ValueError('Error: Must train model before trying to predict any value')

        output = self.predict_fn(self.model.forward(x))
        prediction = torch.max(output, 1)[1]
        return prediction

    def l1_penalty(self):
        return self.l1 * self.get_weights().norm(p=1)

    # TODO: Check this more thoroughly
    def fairness_penalty(self, xi, yi, x, y, s):
        dtype = type(xi.data)
        # If this is the first time calling the penalty, save some info
        if self.fairness_idxs is None:
            self.fairness_idxs = []
            self.vals = []
            self.divisors = []
            for col in s:
                self.vals.append(np.unique(check_and_convert_tensor(x)[:, col]))
                self.fairness_idxs.append([])
                prod = 1
                for val in self.vals[-1]:
                    idxs = (x[:, col] == float(val)).nonzero().squeeze()
                    prod *= len(idxs)
                    self.fairness_idxs[-1].append(idxs)
                self.divisors.append(prod)

        # Actually calculate the penalty
        penalty = 0
        for idxs, val, div in zip(self.fairness_idxs, self.vals, self.divisors):
            # Should be size (self.minibatch_size x n_classes)
            local_dots = torch.mm(xi, self.get_weights())
            for idx in idxs:
                # Because of broadcasting rules, this should create a matrix of absolute differences
                # of size (|S_i: class[idx]| x self.minibatch_size)
                diff = torch.abs(yi.view(1, -1) - y[idx].view(-1, 1)).type(dtype)

                # Size: (|S_i: class[idx]| x n_classes)
                non_local_dot = torch.mm(x[idx], self.get_weights())

                for local_dot in local_dots:
                    # Individual version
                    # Size: |S_i: class[idx]|
                    cross_term = local_dot - non_local_dot ** 2

                    # TODO: Think through this mm call and make sure that it makes sense
                    #       I think that it does... it's summing over all the differences for a 
                    #       given local_dot cross term
                    unsummed_term = diff.t().mm(cross_term)
                    # TODO: Think about why unsummed_term will have dimensions:
                    #       (mini-batch x n_classes)
                    #       The mini-batch part makes sense, but n_classes needs to go somewhere
                    penalty = penalty + (unsummed_term).sum() / div

                    # # Group version
                    # cross_term = cross_term + (local_dot - non_local_dot)
                    # penalty = penalty + ((diff * cross_term).sum() / div) ** 2

        return self.l_fair * penalty

    # # TODO: Incorporate information from s
    # # I think that we will have to store a list of these y_diff matrices based on s
    # def calc_y_diff_mat(self, y, s_idxs):
    #     """Calculates the constant portion of
    #     $\frac{1}{n_1 n_2} \sum_{(x_j, y_j) \in S_i} d(y_1, y_2) (W x_1 - W x_2)^2$ and stores
    #     it in a list of matrices (one for each class pair)
        
    #     Args:
    #         y (LongTensor): Class labels for the problem
    #         s_idxs (list of LongTensor): List containing indices for each y that corresponds to each class label
        
    #     Returns:
    #         dict of FloatTensors: Pairwise absolute differences between class labels
    #     """
    #     if torch.is_tensor(y):
    #         classes = np.unique(y.cpu().numpy())
    #     else:
    #         classes = np.unique(y.data.cpu().numpy())

    #     counts = torch.histc(y.float(), bins=len(classes), min=classes.min(), max=classes.max())
    #     divisor = counts.prod()

    #     # This should broadcast such that y_diff[i, j] = abs(y[i] - y[j]) / divisor
    #     y_diff = torch.abs(y.view(1, -1), y.view(-1, 1)).float() / divisor

    #     return y_diff

    def score(self, x, y):
        prediction = self.predict(x)
        return torch.mean((y == prediction).float()).data[0]

    def get_weights(self):
        if self.model is None:
            raise ValueError('Error: Must train model before asking for weights')
        return self.model.weight

    def set_weights(self, w):
        if torch.is_tensor(w):
            dtype = type(w)
            w_t = torch.nn.Parameter(w.t())
        else:
            dtype = type(w.data)
            w_t = torch.nn.Parameter(w.t().data)

        if self.model is None:
            self.n_features = w.size(0)
            self.n_classes = w.size(1)
            self.model = self.build_model().type(dtype)
        else:
            if self.n_classes != w.size(1) or self.n_features != w.size(0):
                raise ValueError('Given weights matrix shape {} does not match model definition ({}x{})'
                                 .format(w.size(), self.n_classes, self.n_features))
        self.model.weight = w_t

    def get_yhat(self, x):
        if self.model is None:
            raise ValueError('Error: Must train model before asking for yhat')

        return self.predict_fn(self.model.forward(x))
