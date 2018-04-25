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


batch_size = 512
n_epochs = 256

# Import data as pandas dataframes
s_id = ['Sex_Female']
s, x_train, y_train, _, _1 = get_adult_data(s_id)

# # Save the header info before turning into matrices
x_cols = x_train.columns
# y_cols = y_train.columns

# Convert the dataframes into PyTorch variables and cuda-fy if available
x = Variable(torch.from_numpy(x_train.as_matrix()))
y = Variable(torch.from_numpy(y_train.as_matrix()).long())

if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()

# K-fold Cross Validation for FairLogReg fairness search
shared_kwargs = {'ftol': 1e-6, 'n_epochs': n_epochs, 'minibatch_size': batch_size, 'batch_fairness': True}

df_template = {'Type': ['Plain', 'Individual', 'Group'],
               'MSE': [], 'Score': [], 'Group Penalty': [], 'Individual Penalty': [], 'ID_String': [], 'l_fair': None}
penalties = np.logspace(-10, 2, 13)
cv = KFold(3)

models = {}
fold = 0
df = pd.DataFrame()

for train, test in cv.split(x, y):
    train = torch.from_numpy(train).long()
    test = torch.from_numpy(test).long()
    if torch.cuda.is_available():
        train = train.cuda()
        test = test.cuda()
    x_train = x[train].contiguous()
    x_test = x[test].contiguous()
    y_train = y[train].contiguous()
    y_test = y[test].contiguous()

    fold += 1
    print("Fold: {}/3".format(fold))
    for penalty in penalties:
        print("  Penalty: {:0.5g}".format(penalty))
        current_df_dict = copy.deepcopy(df_template)
        penalty = float(penalty)  # PyTorch messes up with numpy types

        # Define models
        plain = FairLogisticRegression(n_epochs=n_epochs, ftol=1e-6, minibatch_size=32)
        indiv = FairLogisticRegression(l_fair=penalty, penalty_type='individual', **shared_kwargs)
        group = FairLogisticRegression(l_fair=penalty, penalty_type='group', **shared_kwargs)

        # Fit them
        plain.fit(x_train, y_train, s)
        indiv.fit(x_train, y_train, s)
        group.fit(x_train, y_train, s)

        # Save the penalty weight
        current_df_dict['l_fair'] = penalty

        # Save string for identifying models
        current_df_dict['ID_String'].extend([plain_str, indiv_str, group_str])

        # Save score info
        plain_score = plain.score(x_test, y_test)
        indiv_score = indiv.score(x_test, y_test)
        group_score = group.score(x_test, y_test)
        current_df_dict['Score'].extend([plain_score, indiv_score, group_score])

        # Save MSE info
        plain_pred = plain.predict(x_test).data.cpu().numpy()
        indiv_pred = indiv.predict(x_test).data.cpu().numpy()
        group_pred = group.predict(x_test).data.cpu().numpy()

        plain_mse = mean_squared_error(plain_pred, y_test.data.cpu().numpy())
        indiv_mse = mean_squared_error(indiv_pred, y_test.data.cpu().numpy())
        group_mse = mean_squared_error(group_pred, y_test.data.cpu().numpy())

        current_df_dict['MSE'].extend([plain_mse, indiv_mse, group_mse])

        # Save Penalty info
        plain_pen_i = 0.
        indiv_pen_i = 0.
        group_pen_i = 0.

        plain_pen_g = 0.
        indiv_pen_g = 0.
        group_pen_g = 0.

        ds = TensorDataset(x_test.data.cpu(), y_test.data.cpu())
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=1)
        for i, data in enumerate(loader):
            inputs, labels = to_Variables(*data)

            plain_pen_i += plain.fairness_penalty(inputs, labels, inputs, labels, s, penalty_type='individual')
            indiv_pen_i += indiv.fairness_penalty(inputs, labels, inputs, labels, s, penalty_type='individual')
            group_pen_i += group.fairness_penalty(inputs, labels, inputs, labels, s, penalty_type='individual')

            plain_pen_g += plain.fairness_penalty(inputs, labels, inputs, labels, s, penalty_type='group')
            indiv_pen_g += indiv.fairness_penalty(inputs, labels, inputs, labels, s, penalty_type='group')
            group_pen_g += group.fairness_penalty(inputs, labels, inputs, labels, s, penalty_type='group')

        plain_pen_i = plain_pen_i.data.cpu().numpy()[0]
        indiv_pen_i = indiv_pen_i.data.cpu().numpy()[0]
        group_pen_i = group_pen_i.data.cpu().numpy()[0]

        plain_pen_g = plain_pen_g.data.cpu().numpy()[0]
        indiv_pen_g = indiv_pen_g.data.cpu().numpy()[0]
        group_pen_g = group_pen_g.data.cpu().numpy()[0]

        current_df_dict['Individual Penalty'].extend([plain_pen_i, indiv_pen_i, group_pen_i])
        current_df_dict['Group Penalty'].extend([plain_pen_g, indiv_pen_g, group_pen_g])

        df = pd.concat([df, pd.DataFrame.from_dict(current_df_dict)])

        df.to_csv('save_temporary_results.csv')

        # Save the models
        plain_str = 'fold: {} pen: {} type: {}'.format(fold, penalty, 'plain')
        indiv_str = 'fold: {} pen: {} type: {}'.format(fold, penalty, 'indiv')
        group_str = 'fold: {} pen: {} type: {}'.format(fold, penalty, 'group')

        plain.model = plain.model.cpu()
        indiv.model = indiv.model.cpu()
        group.model = group.model.cpu()

        models[plain_str] = plain
        models[indiv_str] = indiv
        models[group_str] = group

        with open('save_temporary_models.pkl', 'wb') as f:
            pickle.dump(models, f)
