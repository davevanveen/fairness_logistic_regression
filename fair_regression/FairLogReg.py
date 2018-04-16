import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from cvxpy import *


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


def train_val_split(x, y, val_pct):
    nsamples = len(y)
    nsamples_val = int(val_pct * nsamples)
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


class FairLogisticRegression():
    def __init__(self, lr=0.01, n_classes=None, ftol=1e-9, tolerance_grad=1e-5,
                 fit_intercept=True, n_epochs=32, l_fair=0.0, l2=0.0, l1=0.0,
                 minibatch_size=32, n_jobs=1, validate=0, print_freq=0):
        self.lr = lr
        self.n_classes = n_classes
        self.ftol = ftol
        self.tolerance_grad = tolerance_grad
        self.fit_intercept = fit_intercept
        self.n_epochs = n_epochs
        self.minibatch_size = minibatch_size
        self.n_jobs = n_jobs
        self.verbose = print_freq
        self.validate = validate
        assert validate < 1.0 and validate >= 0.0, "Error: self.validate must be in range [0, 1)"

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

    def fit(self, x, y, s, writer=None): # train model
        # Make sure that s is a list for use in the code below
        if not isinstance(s, list):
            s = [s]

        # Split the data if we are doing validation (i.e. self.validate > 0)
        if self.validate:
            x, y, x_val, y_val = train_val_split(x, y, self.validate)
            old_loss_val = 0
        else:
            x_val = None
            y_val = None
            old_loss_val = None

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

        old_loss = 0
        for epoch in range(self.n_epochs):
            # current_loss = 0
            for i, data in enumerate(loader):
                inputs, labels = to_Variables(*data, cuda=torch.cuda.is_available())

                self.optimizer.zero_grad()

                fx = self.model.forward(inputs)
                loss = self.loss.forward(fx, labels)
                if self.l1:
                    loss += self.l1_penalty()
                if self.l_fair:
                    loss += self.fairness_penalty(inputs, labels, x, y, s)
                loss.backward(retain_graph=True)

                self.optimizer.step()
                self.training_errors_.append(loss.data[0])
                loss_delta = np.abs(old_loss - loss.data[0])
                old_loss = loss.data[0]

            if self.validate:
                fx_val = self.model.forward(x_val)
                loss_val = self.loss.forward(fx_val, y_val)
                if self.l1:
                    loss_val += self.l1_penalty()
                if self.l_fair:
                    loss_val += self.fairness_penalty(x_val, y_val, x_val, y_val, s)
                self.validation_errors_.append(loss_val.data[0])
                loss_delta_val = np.abs(old_loss_val - loss_val.data[0])
                old_loss_val = loss_val.data[0]

            # Report results in a tensorboard logger
            if writer:
                writer.add_scalar('training/CELoss', loss.data[0], epoch)
                writer.add_scalar('training/Accuracy', self.score(x, y), epoch)
                # writer.add_scalar('training/POF', self.pof(x, y), epoch)
                if self.validate:
                    writer.add_scalar('validation/CELoss', loss_val.data[0], epoch)
                    writer.add_scalar('validation/Accuracy', self.score(x_val, y_val), epoch)
                    # writer.add_scalar('validation/POF', self.pof(x_val, y_val), epoch)

            if self.verbose and (epoch + 1) % self.verbose == 0:
                # print('Epoch [{}/{}] Training    CE Loss: {:0.5g}   Accuracy: {:0.5g}   POF: {:0.5g}'
                #       .format(epoch+1, self.n_epochs, loss.data[0], self.score(x, y), self.pof(x, y)))
                # print('              Validation  CE Loss: {:0.5g}   Accuracy: {:0.5g}   POF: {:0.5g}'
                #       .format(epoch+1, self.n_epochs, loss_val.data[0],
                #               self.score(x_val, y_val), self.pof(x_val, y_val)))
                print('Epoch [{}/{}] Training    CE Loss: {:0.5g}   Accuracy: {:0.5g}'
                      .format(epoch+1, self.n_epochs, loss.data[0], self.score(x, y)))
                if self.validate:
                    print('              Validation  CE Loss: {:0.5g}   Accuracy: {:0.5g}'
                          .format(loss_val.data[0], self.score(x_val, y_val)))
            # Check stopping criteria
            if loss_delta < self.ftol:
                break
            if self.validate and loss_delta_val < self.ftol:
                break

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

    def price_of_fairness(self):
    	# This is a meta-optimization problem for comparing fairness penalty across data sets
    	# We just want to see how error varies with changing lambda
    	# Save for possible extension if we choose to look at other data sets
    	lambda = 0
    	lr_reg = FairLogisticRegression()
    	lr_fair = FairLogisticRegression(l_fair = lambda)

        # Framework using cvxpy

    	return pof

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
