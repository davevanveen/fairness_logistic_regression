import torch
from torch import nn

class _LogRegClassifier(nn.Module):
    def __init__(self, input_size, n_classes, bias=True):
        super(_LogRegClassifier, self).__init__()
        self.linear = nn.Linear(input_size, n_classes, bias=bias)
        self.w = self.parameters()

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=1)

class FairLogisticRegression():
    def __init__(self, lr=0.01, n_classes=None, ftol=1e-9, tolerance_grad=1e-5,
                 fit_intercept=True, n_iter=32, lambda_fair=0.0, l2=0.0, l1=0.0):
        self.lr = lr
        self.n_classes = n_classes
        self.ftol = ftol
        self.tolerance_grad = tolerance_grad
        self.fit_intercept = fit_intercept
        self.n_iter = n_iter

        self.lambda_fair = lambda_fair
        self.l2 = l2
        self.l1 = l1

        self.predict_fn = torch.nn.LogSoftmax(dim=1)

        self.model = None
        self.optimizer = None
        self.loss = None
        self.n_features = None
        self.n_samples = None
        self.y_diff = None

    def build_model(self):
        # We don't need the softmax layer here since CrossEntropyLoss already
        # uses it internally.
        model = torch.nn.Linear(int(self.n_features), int(self.n_classes), bias=self.fit_intercept)
        return model

    def fit(self, x, s, y):
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

        def closure():
            self.optimizer.zero_grad()

            fx = self.model.forward(x)
            output = self.loss.forward(fx, y)
            if self.l1:
                output += self.l1_penalty()
            if self.lambda_fair:
                output += self.fairness_penalty(x, s, y)
            output.backward(retain_graph=True)

            return output
        for i in range(self.n_iter):
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

    # TODO
    def fairness_penalty(self, x, s, y):
        if self.y_diff is None:
            self.y_diff = self.calc_y_diff_mat(y, s)
        pass

    # TODO: Incorporate information from s
    # I think that we will have to store a list of these y_diff matrices based on s
    def calc_y_diff_mat(self, y, s):
        if torch.is_tensor(y):
            classes = np.unique(y.cpu().numpy())
        else:
            classes = np.unique(y.data.cpu().numpy())

        counts = torch.histc(y.float(), bins=len(classes), min=classes.min(), max=classes.max())
        divisor = counts.prod()

        # This should broadcast such that y_diff[i, j] = abs(y[i] - y[j]) / divisor
        y_diff = torch.abs(y.view(1, -1), y.view(-1, 1)).float() / divisor

        return y_diff

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
