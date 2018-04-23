import sys
import torch
from torch.autograd import Variable
from DataPreprocessing import get_adult_data
from FairLogReg import FairLogisticRegression
from sklearn.metrics import mean_squared_error
from tensorboardX import SummaryWriter
CUDA_VISIBLE_DEVICES=2  # noqa


def main(s_id, writer_name):
    writer = SummaryWriter(log_dir=writer_name, comment=writer_name)

    # Import data as pandas dataframes
    s, x_train, y_train, x_test, y_test = get_adult_data(s_id)

    # # Save the header info before turning into matrices
    # x_cols = x_train.columns
    # y_cols = y_train.columns

    # Convert the dataframes into PyTorch variables and cuda-fy if available
    x_train = Variable(torch.from_numpy(x_train.as_matrix()))
    y_train = Variable(torch.from_numpy(y_train.as_matrix()).long())
    x_test = Variable(torch.from_numpy(x_test.as_matrix()))

    if torch.cuda.is_available():
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        x_test = x_test.cuda()

    # We'll only compare y_test as a numpy array, so don't bother to convert
    y_test = y_test.as_matrix()

    # Instantiate and fit the model
    # TODO: Add parameterization from files somehow!
    flr = FairLogisticRegression(l_fair=0.1, validate=0.3, print_freq=1,
                                 penalty_type='individual', minibatch_size=512,
                                 batch_fairness=True)
    flr.fit(x_train, y_train, s, writer=writer)

    # Predict x_test, but then convert result to numpy array
    y_pred = flr.predict(x_test).data.cpu().numpy()
    mse = mean_squared_error(y_test, y_pred)

    print('MSE: {}'.format(mse))


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python {} [s_id...] writer_name'.format(sys.argv[0]))
        sys.exit(0)

    main(sys.argv[1:-1], sys.argv[-1])
