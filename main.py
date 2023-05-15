###############################################################################
# General Information
###############################################################################
# Author: Daniel DiPietro | dandipietro.com | https://github.com/dandip

# Original Paper: https://arxiv.org/abs/1912.04871 (Petersen et al)

# main.py: From here, launch deep symbolic regression tasks. All
# hyperparameters are exposed (info on them can be found in train.py). Unless
# you'd like to impose new constraints / make significant modifications,
# modifying this file (and specifically the get_data function) is likely all
# you need to do for a new symbolic regression task.

###############################################################################
# Dependencies
###############################################################################

from train import train
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from utils import generateDataFast, seed_everything


###############################################################################
# Main Function
###############################################################################

# A note on operators:
# Available operators are: '*', '+', '-', '/', '^', 'sin', 'cos', 'tan',
#   'sqrt', 'square', and 'c.' You may also include constant floats, but they
#   must be strings. For variable operators, you must use the prefix var_.
#   Variable should be passed in the order that they appear in your data, i.e.
#   if your input data is structued [[x1, y1] ... [[xn, yn]] with outputs
#   [z1 ... zn], then var_x should precede var_y.


def main(seed=None):

    # 固定seed
    if seed is not None:
        seed_everything(seed)

    # Load training and test data
    X_constants, X_rnn, y_constants, y_rnn = get_data()

    # Perform the regression task
    results = train(
        X_constants,
        y_constants,
        X_rnn,
        y_rnn,
        # operator_list=['*', '+', '-', '/', 'sin', 'cos', 'ln', 'exp', 'var_x1', 'var_x2','c'],
        # operator_list=['+', 'sin', 'cos', 'ln', 'exp', 'var_x'],
        # operator_list=['*', '+', '-', '/', 'sin', 'cos', 'ln', 'exp', 'var_x1', 'var_x2'],
        operator_list=['*', '+', 'sin', 'var_x1'],
        min_length=4,
        max_length=30,
        type='gru',  # 'rnn', 'gru', 'lstm', 'dnc'
        num_layers=1,
        hidden_size=32,
        dropout=0.0,
        lr=0.0005,
        optimizer='adam',
        inner_optimizer='lbfgs',  # 'lbfgs', 'adam', or 'rmsprop'
        inner_lr=0.1,
        inner_num_epochs=25,
        entropy_weight=0.02,  # 论文0.005
        entropy_gamma=0.7,
        risk_factor=0.95,  # ini=0.95
        initial_batch_size=2000,  # ini=2000
        scale_initial_risk=True,
        batch_size=500,
        num_batches=4000,
        use_gpu=False,
        live_print=True,
        summary_print=True,
        config_prior='./config_prior.json',
        reward_type="NRMSE",  # 'MSE', 'NRMSE', 'MEDL', 'Pareto_optimal'
        Pareto_frontier=False  # compute pareto frontier
    )

    # Unpack results
    epoch_best_rewards = results[0]
    epoch_best_expressions = results[1]
    best_reward = results[2]
    best_expression = results[3]
    epoch_mean_rewards = results[4]  # 整个epoch的平均reward
    # print("epoch best reward: ", epoch_best_rewards)

    # Plot best rewards each epoch
    plt.plot([i + 1 for i in range(len(epoch_best_rewards))], epoch_best_rewards)  # best reward of full epoch
    # plt.plot([i + 1 for i in range(len(epoch_mean_rewards))], epoch_mean_rewards)  # mean reward of full epoch
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.title('Reward over Time')
    plt.show()


###############################################################################
# Getting Data
###############################################################################

def get_data():
    """Constructs data for model (currently x^3 + x^2 + x)
    """

    # TODO:Nguyen
    # target_eq = Nguyen_1 = "x_1**3 + x_1**2 + x_1"  # seed=43 122epochs
    # target_eq = Nguyen_2 = "x_1**4 + x_1**3 + x_1**2 + x_1"  #
    # target_eq = Nguyen_3 = "x_1**5 + x_1**4 + x_1**3 + x_1**2 + x_1"  # yes
    # target_eq = Nguyen_4 = "x_1**6 + x_1**5 + x_1**4 + x_1**3 + x_1**2 + x_1"  # yes 551epochs
    # target_eq = Nguyen_5 = "sin(x_1**2)*cos(x_1) - 1"  # yes 135epochs
    # target_eq = Nguyen_6 = "sin(x_1) + sin(x_1 + x_1**2)"  # yes 151epochs
    # target_eq = Nguyen_7 = "log(x_1+1)+log(x_1**2+1)"  # yes 138epochs
    # target_eq = Nguyen_8 = "x_1**0.5"  # yes 827epochs
    # target_eq = Nguyen_9 = "sin(x_1) + sin(x_2**2)"  # yes 189 epochs
    # target_eq = Nguyen_10 = "2*sin(x_1)*cos(x_2)"  # yes 459 epochs
    # target_eq = Nguyen_11 = "x_1**x_2"  # exp(ln(x1) * x2) 143 epochs
    # target_eq = Nguyen_12 = "x_1**4-x_1**3+0.5*x_2**2-x_2"  # No
    target_eq = "x_1**5 + x_1**4"  # No

    # target_eq = "sin(x_1**2 + x_2**2)"
    # target_eq = "sin(x_1)+cos(x_1)"  # yes
    # target_eq = "x_1 + sin(x_1) + cos(x_1)"

    # generate data
    n_ponits = 100
    n_vars = 1
    min_x = 0
    max_x = 10
    X, y = generateDataFast(target_eq, n_points=n_ponits, n_vars=n_vars, decimals=8, min_x=min_x, max_x=max_x)
    if len(y) == 0:
        print("y value is None !")
        exit()
    X, y = np.array(X), np.array(y)

    # Split randomly
    comb = list(zip(X, y))
    random.shuffle(comb)
    X, y = zip(*comb)

    # Proportion used to train constants versus benchmarking functions
    # training_proportion = 0.2
    # div = int(training_proportion * len(X))
    # X_constants, X_rnn = np.array(X[:div]), np.array(X[div:])
    # y_constants, y_rnn = np.array(y[:div]), np.array(y[div:])
    X_constants, X_rnn = np.array(X), np.array(X)
    y_constants, y_rnn = np.array(y), np.array(y)
    X_constants, X_rnn = torch.Tensor(X_constants), torch.Tensor(X_rnn)
    y_constants, y_rnn = torch.Tensor(y_constants), torch.Tensor(y_rnn)
    return X_constants, X_rnn, y_constants, y_rnn


if __name__ == '__main__':
    main(seed=59)
