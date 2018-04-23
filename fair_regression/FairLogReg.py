import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from timeit import default_timer as timer
# from cvxpy import *


def to_Variables(*args):
    ret = []
    for arg in args:
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


def create_param_dict(fairLogReg):
    d = {}
    d['lr'] = fairLogReg.lr
    d['n_classes'] = fairLogReg.n_classes
    d['ftol'] = fairLogReg.ftol
    d['tolerance_grad'] = fairLogReg.tolerance_grad
    d['n_epochs'] = fairLogReg.n_epochs
    d['minibatch_size'] = fairLogReg.minibatch_size
    d['n_jobs'] = fairLogReg.n_jobs
    d['validate'] = fairLogReg.validate
    d['print_freq'] = fairLogReg.verbose
    d['penalty_type'] = fairLogReg.penalty_type

    return d


class FairLogisticRegression():
    def __init__(self, lr=0.01, n_classes=None, ftol=1e-9, tolerance_grad=1e-5,
                 fit_intercept=True, n_epochs=32, l_fair=0.0, l1=0.0, l2=0.0,
                 minibatch_size=32, n_jobs=1, validate=0, print_freq=0,
                 penalty_type='individual'):
        self.lr = lr
        self.n_classes = n_classes
        self.ftol = ftol
        self.tolerance_grad = tolerance_grad
        self.fit_intercept = fit_intercept
        self.n_epochs = n_epochs
        self.minibatch_size = minibatch_size
        self.n_jobs = n_jobs
        self.verbose = print_freq
        self.penalty_type = penalty_type
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

        self.training_errors_ = []
        self.validation_errors_ = []

    def build_model(self):
        # We don't need the softmax layer here since CrossEntropyLoss already
        # uses it internally.
        model = torch.nn.Linear(int(self.n_features), int(self.n_classes), bias=self.fit_intercept)
        return model

    def fit(self, x, y, s, writer=None):  # train model
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

        # Create a placeholder variable for saved index information
        train_idx_info = None
        val_idx_info = None

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
            train_idx_info = None
            for i, data in enumerate(loader):
                inputs, labels = to_Variables(*data)

                self.optimizer.zero_grad()

                fx = self.model.forward(inputs)
                loss = self.loss.forward(fx, labels)
                if self.l1:
                    loss += self.l1 * self.l1_penalty()
                if self.l_fair:
                    fp, train_idx_info = self.fairness_penalty2(inputs, labels, x, y, s,
                                                                penalty_type=self.penalty_type,
                                                                saved_idx_info=train_idx_info)
                    loss += self.l_fair * fp
                loss.backward(retain_graph=True)

                self.optimizer.step()
                self.training_errors_.append(loss.data[0])
                loss_delta = np.abs(old_loss - loss.data[0])
                old_loss = loss.data[0]

            if self.validate:
                fx_val = self.model.forward(x_val)
                loss_val = self.loss.forward(fx_val, y_val)

                if self.l1:
                    loss_val += self.l1 * self.l1_penalty()
                if self.l_fair:
                    fp_val, val_idx_info = self.fairness_penalty2(x_val, y_val, x_val, y_val, s,
                                                                  penalty_type=self.penalty_type,
                                                                  saved_idx_info=val_idx_info)
                    loss_val += self.l_fair * fp_val
                self.validation_errors_.append(loss_val.data[0])
                loss_delta_val = np.abs(old_loss_val - loss_val.data[0])
                old_loss_val = loss_val.data[0]

            # Report results in a tensorboard logger
            if writer:
                writer.add_scalar('training/CELoss', loss.data[0], epoch)
                writer.add_scalar('training/Accuracy', self.score(x, y), epoch)
                if self.l_fair:
                    writer.add_scalar('training/fairness_penalty', fp, epoch)
                if self.validate:
                    writer.add_scalar('validation/CELoss', loss_val.data[0], epoch)
                    writer.add_scalar('validation/Accuracy', self.score(x_val, y_val), epoch)
                    if self.l_fair:
                        writer.add_scalar('validation/fairness_penalty', fp_val, epoch)

            if self.verbose and (epoch + 1) % self.verbose == 0:
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
        return self.get_weights().norm(p=1)

    # TODO: Check this more thoroughly
    def fairness_penalty(self, xi, yi, x, y, s, penalty_type='individual', saved_idx_info=None):
        dtype = type(xi.data)
        # If this is the first time calling the penalty, save some info
        if saved_idx_info is None:
            fairness_idxs = []
            vals = []
            divisors = []
            for col in s:
                vals.append(np.unique(check_and_convert_tensor(x)[:, col]))
                fairness_idxs.append([])
                prod = 1
                for val in vals[-1]:
                    idxs = (x[:, col] == float(val)).nonzero().squeeze()
                    prod *= len(idxs)
                    fairness_idxs[-1].append(idxs)
                divisors.append(float(prod))
            saved_idx_info = list(zip(fairness_idxs, vals, divisors))

        # Actually calculate the penalty
        penalty = 0
        for idxs, val, div in saved_idx_info:
            # Should be size (self.minibatch_size x n_classes)
            local_dots = torch.mm(xi, self.get_weights().t())
            for idx in idxs:
                # Because of broadcasting rules, this should create a matrix of absolute differences
                # of size (|S_i: class[idx]| x self.minibatch_size)
                diff = torch.abs(yi.view(1, -1) - y[idx].view(-1, 1)).type(dtype)

                # Size: (|S_i: class[idx]| x n_classes)
                non_local_dot = torch.mm(x[idx], self.get_weights().t())

                for local_dot in local_dots:
                    # Individual version
                    # Size: |S_i: class[idx]|
                    cross_term = (local_dot - non_local_dot) ** 2

                    if penalty_type == 'individual':
                        # TODO: Think through this mm call and make sure that it makes sense
                        #       I think that it does... it's summing over all the differences for a
                        #       given local_dot cross term
                        unsummed_term = diff.t().mm(cross_term)
                        # TODO: Think about why unsummed_term will have dimensions:
                        #       (mini-batch x n_classes)
                        #       The mini-batch part makes sense, but n_classes needs to go somewhere
                        penalty = penalty + (unsummed_term).sum() / div
                    elif penalty_type == 'group':
                        # Group version
                        cross_term = (local_dot - non_local_dot)
                        penalty = penalty + ((diff * cross_term).sum() / div) ** 2

        return penalty, saved_idx_info

    def price_of_fairness(self, alpha, xi, yi, x, y, s):
        # overall: need to get four terms: l(w), l(w*), f(w), f(w*)
        # easy to get: l(w*), f(w*) b/c they are values. w* minimizes loss term, i.e. lambda=0
        # must backprop through: l(w), f(w) b/c we're minimizing w.r.t. w

        # Q'n: is w = optimized weights when the model is trained WITH fairness regularizer?

        w_star = self.get_weights()

        inputs, labels = to_Variables(*data)
        # self.optimizer.zero_grad()
        fx = self.model.forward(inputs)
        loss = self.loss.forward(fx, labels)

        # Framework using cvxpy
        constraint = f(w) <= alpha*f(w_star)
        obj = Minimize(l(w) / l(w_star))
        prob = Problem(obj, constraint)
        pof = prob.solve()  # will pof be an attribute of the model? I don't think so

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

    def save(self, fn):
        new_dict = self.model.state_dict().copy()
        new_dict['validation_errors_'] = self.validation_errors_
        new_dict['training_errors_'] = self.training_errors_
        new_dict['parameters'] = create_param_dict(self)

        torch.save(new_dict, fn)

    @classmethod
    def load(cls, fn):
        state_dict = torch.load(fn)

        # Instatiate self as the class
        self = cls(**state_dict['parameters'])

        # Get non-parameter readings out of the state dictionary
        self.validation_errors_ = state_dict['validation_errors_']
        self.training_errors_ = state_dict['training_errors_']

        # Get rid of non parameters in a copy of the dict
        new_dict = state_dict.copy()
        for name in state_dict.keys():
            if name not in ['weight', 'bias']:
                del new_dict[name]

        # Instantiate the model
        self.n_features = state_dict['weight'].size(1)
        self.n_classes = state_dict['weight'].size(0)
        self.model = self.build_model().type(type(new_dict['weight']))
        self.model.load_state_dict(new_dict)

        return self

    def fairness_penalty2(self, xi, yi, x, y, s, penalty_type='individual', saved_idx_info=None):
        dtype = type(xi.data)
        # If this is the first time calling the penalty, save some info
        if saved_idx_info is None:
            fairness_idxs = []
            vals = []
            divisors = []
            for col in s:
                vals.append(np.unique(check_and_convert_tensor(x)[:, col]))
                fairness_idxs.append([])
                prod = 1
                for val in vals[-1]:
                    idxs = (x[:, col] == float(val)).nonzero().squeeze()
                    prod *= len(idxs)
                    fairness_idxs[-1].append(idxs)
                divisors.append(float(prod))
            saved_idx_info = list(zip(fairness_idxs, vals, divisors))

        # We're going to be indexing into the prediction in the loops below
        y_soft_pred = self.model.forward(x)
        yi_soft_pred = self.model.forward(xi)

        # Actually calculate the penalty
        penalty = 0
        for idxs, val, div in saved_idx_info:
            for idx in idxs:
                # Because of broadcasting rules, this should create a matrix of absolute differences
                # of size (|S_i: class[idx]| x self.minibatch_size)
                y_diff = torch.abs(yi.view(1, -1) - y[idx].view(-1, 1)).type(dtype)

                # For multinomial models, sum over each possible output class separately
                for class_idx in range(y_soft_pred.size(1)):
                    yi_soft_pred_col = yi_soft_pred[:, class_idx].contiguous()
                    y_soft_pred_col = y_soft_pred[idx][:, class_idx].contiguous()
                    pred_diff = (yi_soft_pred_col.view(1, -1) - y_soft_pred_col.view(-1, 1)).type(dtype)

                    if penalty_type == 'individual':
                        # Individual version
                        pred_diff = pred_diff ** 2
                        unsummed_term = y_diff * pred_diff
                        penalty = penalty + (unsummed_term).sum() / div
                    elif penalty_type == 'group':
                        # Group version
                        unsummed_term = y_diff * pred_diff
                        penalty = penalty + (unsummed_term.sum() / div) ** 2

        return penalty, saved_idx_info
