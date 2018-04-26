# Open the models and pick one or more to show the more fair version of the plots in the presentation
import sys
sys.path.append('../fair_regression')

import torch
import pickle
import pandas as pd

from torch.autograd import Variable
from DataPreprocessing import get_adult_data


s, _, _1, x_test, y_test = get_adult_data(['Sex_Female'])
with open('../fair_regression/save_temporary_models.pkl', 'rb') as f:
    model_dict = pickle.load(f)

torch_xt = Variable(torch.from_numpy(x_test.as_matrix()))
if torch.cuda.is_available():
    torch_xt = torch_xt.cuda()

# Pick out models from fold 1 (doesn't really matter which one)
model_str = 'fold: 1 pen: {:0.5g} type: {}'

model_types = ['plain', 'indiv', 'group']
penalties = [1e-3, 1e-2, 1e-1]

x_test['True Y'] = y_test

new_df = pd.DataFrame()
for t in model_types:
    for p in penalties:
        unique_id = model_str.format(p, t)
        my_model = model_dict[unique_id].cuda()

        pred = my_model.predict(torch_xt).data.cpu().numpy()
        x_test[unique_id] = pred

x_test.to_csv('test_predictions.csv')
