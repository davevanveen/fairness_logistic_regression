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
penalties = np.logspace(-10, 2, 7)
cv = KFold(3)

models = {}
fold = 0
df = pd.DataFrame()

for train, test in cv.split(x, y):
    # Convert all of the arrays into PyTorch Arrays
    train = torch.from_numpy(train).long()
    test = torch.from_numpy(test).long()
    if torch.cuda.is_available():
        train = train.cuda()
        test = test.cuda()

    # Make the arrays contiguous for more spatial locality
    x_train = x[train].contiguous()
    x_test = x[test].contiguous()
    y_train = y[train].contiguous()
    y_test = y[test].contiguous()

    ds = TensorDataset(x_test.data.cpu(), y_test.data.cpu())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=1)

    fold += 1
    print("Fold: {}/3".format(fold))
    for penalty in penalties:
        print("  Penalty: {:0.5g}".format(penalty))
        current_df_dict = copy.deepcopy(df_template)
        current_df_dict['l_fair'] = penalty  # Save the penalty weight
        penalty = float(penalty)  # PyTorch messes up with numpy types

        for pen_type in ['plain', 'individual, group']:
            # Define model
            if pen_type == 'plain':
                model = FairLogisticRegression(n_epochs=n_epochs, ftol=1e-6, minibatch_size=32)
            else:
                model = FairLogisticRegression(l_fair=penalty, penalty_type=pen_type, **shared_kwargs)

            # Fit it
            model.fit(x_train, y_train, s)

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

                pen_i += model.fairness_penalty(inputs, labels, inputs, labels, s, penalty_type='individual')
                pen_g += model.fairness_penalty(inputs, labels, inputs, labels, s, penalty_type='group')

            # Convert penalties back into numpy for saving them
            pen_i = pen_i.data.cpu().numpy()[0]
            pen_g = pen_g.data.cpu().numpy()[0]

            current_df_dict['Individual Penalty'].append(pen_i)
            current_df_dict['Group Penalty'].append(pen_g)

            model.model = model.model.cpu()
            models[save_str] = model

        # Concatenate the current dictionary to a dataframe and save its output
        df = pd.concat([df, pd.DataFrame.from_dict(current_df_dict)])
        df.to_csv('save_temporary_results.csv')

        with open('save_temporary_models.pkl', 'wb') as f:
            pickle.dump(models, f)
