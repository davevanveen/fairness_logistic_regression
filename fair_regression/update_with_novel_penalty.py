import sys
import json
import torch
import pickle
import os.path
import pandas as pd

from glob import glob
from torch.autograd import Variable
from FairLogReg import to_Variables
from DataPreprocessing import get_adult_data
from torch.utils.data import DataLoader, TensorDataset


def update_df(json_fn, pkl_fn, csv_fn):
    with open(json_fn, 'r') as json_file:
        my_kws = json.load(json_file)
    with open(pkl_fn, 'rb') as pickle_file:
        models = pickle.load(pickle_file)

    df = pd.read_csv(csv_fn)

    # Load and convert the data
    s, _, _1, x_test, y_test = get_adult_data(my_kws['s_ids'])
    x_test = Variable(torch.from_numpy(x_test.as_matrix()))
    y_test = Variable(torch.from_numpy(y_test.as_matrix()).long())

    # Create a dataloader to iterate over minibatches
    ds = TensorDataset(x_test.data.cpu(), y_test.data.cpu())
    loader = DataLoader(ds, batch_size=my_kws['batch_size'], shuffle=False)

    current_df_novel_penalties = []

    for idx, row in df.iterrows():
        # Get the current model
        current_model = models[row.ID_String]
        if torch.cuda.is_available():
            current_model.model = current_model.model.cuda()

        # Calculate the penalty
        penalty = 0.
        # Load across batches for computational efficiency
        for i, data in enumerate(loader):
            inputs, labels = to_Variables(*data)
            penalty += current_model.fairness_penalty(inputs, labels, inputs, labels, s, penalty_type='novel')

        # Add the value to the current list
        current_df_novel_penalties.append(penalty.data[0])

    # Now add the complete column to the current dataframe and write it to a file
    df['Novel Penalty'] = current_df_novel_penalties
    df.to_csv(csv_fn[:-4] + '_updated.csv')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python {} directory_to_upate'.format(sys.argv[0]))
        sys.exit(0)
    update_dir = sys.argv[1]

    # Make sure the directory string ends in a '/' character
    if update_dir[-1] != '/':
        update_dir += '/'

    # Get a list of all csv files
    unique_file_ids = glob('{}*.csv'.format(update_dir))

    # Remove the updated csv_files from this list
    unique_file_ids = [x for x in unique_file_ids if '_updated.csv' not in x]

    if not unique_file_ids:
        raise FileNotFoundError('Error: No csv files found in {}'.format(update_dir))

    # Iterate over all of the csv files
    for i, unid in enumerate(unique_file_ids):
        unid = unid[:-4]  # Grab everything but the '.csv' part

        # Update the user as to the progress of the code
        print('[{}/{}]   Updating unique ID: {}'.format(i+1, len(unique_file_ids), unid))

        # Form the other filenames
        json_fn = unid + '.json'
        pkl_fn = unid + '.pkl'
        csv_fn = unid + '.csv'

        # Check that they all exist
        if not os.path.isfile(csv_fn):
            raise FileNotFoundError('Error: {} found by glob, but does not exist'.format(csv_fn))
        fmt = 'Error: {csv} exists, but {{other}} does not'.format(csv=csv_fn)
        if not os.path.isfile(json_fn):
            raise FileNotFoundError(fmt.format(other=json_fn))
        if not os.path.isfile(pkl_fn):
            raise FileNotFoundError(fmt.format(other=pkl_fn))

        update_df(json_fn, pkl_fn, csv_fn)
