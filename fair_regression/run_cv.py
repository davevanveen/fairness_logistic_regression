import sys
import json
import copy
import torch
from torch.autograd import Variable

from FairLogReg import FairLogisticRegression
from DataPreprocessing import get_adult_data
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import pickle


def to_Variables(*args):
    ret = []
    for arg in args:
        if torch.cuda.is_available():
            ret.append(Variable(arg).cuda())
        else:
            ret.append(Variable(arg))

    return ret


class RunFairCV():
    def __init__(self, s_ids=['Sex_Female'], n_epochs=256, batch_size=512, plain_batch_size=32, ftol=1e-6,
                 batch_fairness=True, l_fair_logspace=[-6, 2, 4], cv=3, csv_fn='save_temporary_results.csv',
                 models_fn='save_temporary_models.pkl', pred_fn='save_temporary_preds.csv'):
        # Load data based on inputs
        self.s, x_train, y_train, self.x_test, y_test = get_adult_data(s_ids)

        # Convert the dataframes into PyTorch variables and cuda-fy if available
        x = Variable(torch.from_numpy(x_train.as_matrix()))
        y = Variable(torch.from_numpy(y_train.as_matrix()).long())
        xt = Variable(torch.from_numpy(self.x_test.as_matrix()))

        if torch.cuda.is_available():
            self.x = x.cuda()
            self.y = y.cuda()
            self.xt = xt.cuda()

        # Add the true values to x_test for later analysis
        self.x_test['Y True'] = y_test

        # K-fold Cross Validation for FairLogReg fairness search
        self.shared_kwargs = {'ftol': ftol, 'n_epochs': n_epochs,
                              'minibatch_size': batch_size,
                              'batch_fairness': batch_fairness}

        self.df_template = {'Type': [], 'MSE': [], 'Score': [], 'Group Penalty': [],
                            'Individual Penalty': [], 'ID_String': [], 'l_fair': None}
        self.penalties = np.logspace(*l_fair_logspace)
        self.splits = cv
        if cv > 1:
            self.cv = KFold(cv)
        else:
            self.cv = None

        self.models = {}

        self.batch_size = batch_size
        self.plain_batch_size = plain_batch_size
        self.ftol = ftol
        self.n_epochs = n_epochs
        self.batch_fairness = batch_fairness
        self.csv_fn = csv_fn
        self.models_fn = models_fn
        self.pred_fn = pred_fn

    def run(self):
        fold = 0
        df = pd.DataFrame()

        # Make sure that we can work with one fold if we want to
        if self.splits > 1:
            splitter = self.cv.split(self.x, self.y)
        else:
            splitter = [[np.arange(len(self.x)), np.arange(len(self.y))]]

        for train, test in splitter:
            # Convert all of the arrays into PyTorch Arrays
            train = torch.from_numpy(train).long()
            test = torch.from_numpy(test).long()
            if torch.cuda.is_available():
                train = train.cuda()
                test = test.cuda()

            # Make the arrays contiguous for more spatial locality
            x_train = self.x[train].contiguous()
            x_test = self.x[test].contiguous()
            y_train = self.y[train].contiguous()
            y_test = self.y[test].contiguous()

            ds = TensorDataset(x_test.data.cpu(), y_test.data.cpu())
            loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=1)

            fold += 1
            print("Fold: {}/{}".format(fold, self.splits))
            for penalty in self.penalties:
                print("  Penalty: {:0.5g}".format(penalty))
                current_df_dict = copy.deepcopy(self.df_template)
                current_df_dict['l_fair'] = penalty  # Save the penalty weight
                penalty = float(penalty)  # PyTorch messes up with numpy types

                for pen_type in ['plain', 'individual', 'group', 'novel']:
                    current_df_dict['Type'].append(pen_type.title())
                    # Define model
                    if pen_type == 'plain':
                        model = FairLogisticRegression(n_epochs=self.n_epochs, ftol=self.ftol,
                                                       minibatch_size=self.plain_batch_size)
                    else:
                        model = FairLogisticRegression(l_fair=penalty, penalty_type=pen_type, **self.shared_kwargs)

                    # Fit it
                    model.fit(x_train, y_train, self.s)

                    # Save string for identifying models
                    save_str = 'fold: {} pen: {} type: {}'.format(fold, penalty, pen_type[:5])
                    current_df_dict['ID_String'].append(save_str)

                    # Save score info
                    score = model.score(x_test, y_test)
                    current_df_dict['Score'].append(score)

                    # Save MSE info
                    pred = model.predict(x_test).data.cpu().numpy()
                    mse = mean_squared_error(pred, y_test.data.cpu().numpy())
                    current_df_dict['MSE'].append(mse)

                    # Save Penalty info
                    pen_i = 0.
                    pen_g = 0.

                    # Calculate the penalty measures in minibatches for the given train/test split, penalty, and model
                    for i, data in enumerate(loader):
                        inputs, labels = to_Variables(*data)

                        pen_i += model.fairness_penalty(inputs, labels, inputs, labels, self.s, penalty_type='individual')  # noqa
                        pen_g += model.fairness_penalty(inputs, labels, inputs, labels, self.s, penalty_type='group')

                    # Convert penalties back into numpy for saving them
                    pen_i = pen_i.data.cpu().numpy()[0]
                    pen_g = pen_g.data.cpu().numpy()[0]

                    current_df_dict['Individual Penalty'].append(pen_i)
                    current_df_dict['Group Penalty'].append(pen_g)

                    # Predict test data
                    pred = model.predict(self.xt).data.cpu().numpy()
                    self.x_test[save_str] = pred

                    model.model = model.model.cpu()
                    self.models[save_str] = model

                # Concatenate the current dictionary to a dataframe and save its output
                df = pd.concat([df, pd.DataFrame.from_dict(current_df_dict)])
                df.to_csv(self.csv_fn)
                self.df = df

                # Save prediction info
                self.x_test.to_csv(self.pred_fn)

                with open(self.models_fn, 'wb') as f:
                    pickle.dump(self.models, f)


if __name__ == '__main__':
    if len(sys.argv) > 2:
        print('Usage: python {} [json_filename]'.format(sys.argv[0]))
        sys.exit(0)

    # Load JSON file if present
    if len(sys.argv) == 2:
        json_fn = sys.argv[1]
        with open(json_fn, 'r') as json_file:
            my_kws = json.load(json_file)
    else:
        my_kws = {}

    # Make sure all arguments are valid arguments
    legal_kws = ['s_ids', 'n_epochs', 'batch_size', 'plain_batch_size', 'ftol',
                 'batch_fairness', 'l_fair_logspace', 'cv', 'csv_fn', 'models_fn',
                 'pred_fn']
    for kw in my_kws.keys():
        illegal_kws = []
        if kw not in legal_kws:
            illegal_kws.append(kw)
    if len(illegal_kws) > 0:
        print('Error: Illegal keywords found in {}: {}'.format(json_fn, illegal_kws))
        sys.exit(1)

    # Actually run the cross validation model
    my_cv = RunFairCV(**my_kws)
    my_cv.run()
