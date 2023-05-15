###############################################################################
# General Information
###############################################################################
# Author: Daniel DiPietro | dandipietro.com | https://github.com/dandip

# Original Paper: https://arxiv.org/abs/1912.04871 (Petersen et al)

# train.py: Contains main training loop (and reward functions) for PyTorch
# implementation of Deep Symbolic Regression.

###############################################################################
# Dependencies
###############################################################################

import time
import random

import omegaconf
import torch
import torch.nn as nn
import numpy as np
import sympy as sp
from torch.autograd import Variable
from operators import Operators
from prior import make_prior
from rnn import DSRRNN
from expression_utils import *
from collections import Counter
from utils import load_config, benchmark, description_length_complexity
from get_pareto import ParetoSet, Point

###############################################################################
# Main Training loop
###############################################################################

def train(
        X_constants,
        y_constants,
        X_rnn,
        y_rnn,
        operator_list=['*', '+', '-', '/', '^', 'cos', 'sin', 'c', 'var_x'],
        min_length=2,
        max_length=12,
        type='gru',
        num_layers=1,
        dropout=0.0,
        lr=0.0005,
        optimizer='adam',
        inner_optimizer='rmsprop',
        inner_lr=0.1,
        inner_num_epochs=15,
        entropy_weight=0.03,
        entropy_gamma=0.7,
        risk_factor=0.95,
        initial_batch_size=2000,
        scale_initial_risk=True,
        batch_size=500,
        num_batches=200,
        hidden_size=500,
        use_gpu=False,
        live_print=True,
        summary_print=True,
        config_prior=None,
        reward_type=None,
        Pareto_frontier=False

         ):
    """Deep Symbolic Regression Training Loop

    ~ Parameters ~
    - X_constants (Tensor): X dataset used for training constants
    - y_constants (Tensor): y dataset used for training constants
    - X_rnn (Tensor): X dataset used for obtaining reward / training RNN
    - y_rnn (Tensor): y dataset used for obtaining reward / training RNN
    - operator_list (list of str): operators to use (all variables must have prefix var_)
    - min_length (int): minimum number of operators to allow in expression
    - max_length (int): maximum number of operators to allow in expression
    - type ('rnn', 'lstm', or 'gru'): type of architecture to use
    - num_layers (int): number of layers in RNN architecture
    - dropout (float): dropout (if any) for RNN architecture
    - lr (float): learning rate for RNN
    - optimizer ('adam' or 'rmsprop'): optimizer for RNN
    - inner_optimizer ('lbfgs', 'adam', or 'rmsprop'): optimizer for expressions
    - inner_lr (float): learning rate for constant optimization
    - inner_num_epochs (int): number of epochs for constant optimization
    - entropy_weight (float): entropy weight for RNN
    - risk_factor (float, >0, <1): we discard the bottom risk_factor quantile
      when training the RNN
    - batch_size (int): batch size for training the RNN
    - num_batches (int): number of batches (will stop early if found)
    - hidden_size (int): hidden dimension size for RNN
    - use_gpu (bool): whether or not to train with GPU
    - live_print (bool): if true, will print updates during training process

    ~ Returns ~
    A list of four lists:
    [0] epoch_best_rewards (list of float): list of highest reward obtained each epoch
    [1] epoch_best_expressions (list of Expression): list of best expression each epoch
    [2] best_reward (float): best reward obtained
    [3] best_expression (Expression): best expression obtained
    """

    if Pareto_frontier:
        PA = ParetoSet()

    epoch_best_rewards = []
    epoch_mean_rewards = []
    epoch_best_expressions = []
    epoch_mean_length = []
    epoch_mean_dl = []

    # Establish GPU device if necessary
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # load config prior
    config_prior = load_config(config_path=config_prior)["prior"]

    # Initialize operators, RNN, and optimizer
    operators = Operators(operator_list, device)
    prior = make_prior(library=operators, config_prior=config_prior)
    dsr_rnn = DSRRNN(operators, hidden_size, device, min_length=min_length,
                     max_length=max_length, type=type, dropout=dropout, prior=prior, entropy_gamma=entropy_gamma).to(device)

    if optimizer == 'adam':
        optim = torch.optim.Adam(dsr_rnn.parameters(), lr=lr)
    else:
        optim = torch.optim.RMSprop(dsr_rnn.parameters(), lr=lr)

    # Best expression and its performance
    best_expression, best_performance = None, float('-inf')

    # First sampling done outside of loop for initial batch size if desired
    start = time.time()
    # sequences, sequence_lengths, log_probabilities, entropies = dsr_rnn.sample_sequence(initial_batch_size)
    sequences, sequence_lengths, log_probabilities, entropies = dsr_rnn.sample_n_expressions(initial_batch_size)  # 重写的采样过程，新增了各种约束
    for i in range(num_batches):

        expr_length_batch = []
        complexity_batch = []  # DL

        # Convert sequences into Pytorch expressions that can be evaluated
        expressions = []
        for j in range(len(sequences)):
            expressions.append(
                Expression(operators, sequences[j].long().tolist(), sequence_lengths[j].long().tolist()).to(device)
            )

        # 计算epoch的平均表达式长度
        epoch_mean_length.append(torch.mean(sequence_lengths.float()).item())

        # Optimize constants of expressions (training data)
        optimize_constants(expressions, X_constants, y_constants, inner_lr, inner_num_epochs, inner_optimizer)

        # Benchmark expressions (test dataset)
        rewards = []
        for expression in expressions:
            rewards.append(benchmark(expression, X_rnn, y_rnn, reward_type))
        rewards = torch.tensor(rewards)

        # 帕累托前沿
        if Pareto_frontier:
            for j, expression in enumerate(expressions):
                PA.add(Point(x=description_length_complexity(str(expression)),
                             y=rewards[j].item(),
                             data=sp.sympify(str(expression))))

        # Update best expression
        best_epoch_expression = expressions[np.argmax(rewards)]
        epoch_best_expressions.append(best_epoch_expression)
        epoch_best_rewards.append(max(rewards).item())
        epoch_mean_rewards.append(torch.mean(rewards).item())
        if max(rewards) > best_performance:
            best_performance = max(rewards)
            best_expression = best_epoch_expression

        # Early stopping criteria
        # if reward_type == 'MSE' or reward_type == 'NRMSE':
        #     if best_performance >= 0.999999:
        #         best_str = str(best_expression)
        #         if live_print:
        #             print("~ Early Stopping Met ~")
        #             print(f"""Best Expression: {sp.sympify(best_str)}""")
        #         break
        # if reward_type == "MEDL" or reward_type == "Pareto_optimal":
            # 因为描述长度对误差更敏感，极小的残差其描述长度并不小，所以使用R方来评估来决定是否早停
        best_R_2 = benchmark(best_expression, X_rnn, y_rnn, reward_type="R^2")
        print(best_R_2)
        if best_R_2 >= 0.99:
            best_str = str(best_expression)
            if live_print:
                print("~ Early Stopping Met ~")
                print("Total mean length: ", np.mean(epoch_mean_length))
                print("Best R^2: ", best_R_2)
                print(f"""Best Expression: {sp.sympify(best_str)}""")
            break

        # Compute risk threshold
        if i == 0 and scale_initial_risk:
            threshold = np.quantile(rewards, 1 - (1 - risk_factor) / (initial_batch_size / batch_size))
            # print("first threshold: ", threshold)
        else:
            threshold = np.quantile(rewards, risk_factor)
        indices_to_keep = torch.tensor([j for j in range(len(rewards)) if rewards[j] > threshold])

        if len(indices_to_keep) == 0 and summary_print:
            print("Threshold removes all expressions. Terminating.")
            break

        # Select corresponding subset of rewards, log_probabilities, and entropies
        rewards = torch.index_select(rewards, 0, indices_to_keep)
        log_probabilities = torch.index_select(log_probabilities, 0, indices_to_keep)
        entropies = torch.index_select(entropies, 0, indices_to_keep)

        # Compute risk seeking and entropy gradient
        risk_seeking_grad = torch.sum((rewards - threshold) * log_probabilities, axis=0)
        entropy_grad = torch.sum(entropies, axis=0)

        # Mean reduction and clip to limit exploding gradients
        risk_seeking_grad = torch.clip(risk_seeking_grad / len(rewards), -1e6, 1e6)
        entropy_grad = entropy_weight * torch.clip(entropy_grad / len(rewards), -1e6, 1e6)

        # Compute loss and backpropagate
        loss = -1 * lr * (risk_seeking_grad + entropy_grad)
        loss.backward()
        optim.step()

        # Epoch Summary
        if live_print:
            print(f"""Epoch: {i + 1} ({round(float(time.time() - start), 2)}s elapsed)
            Total mean length: {np.mean(epoch_mean_length)}
            Entropy Loss: {entropy_grad.item()}
            Risk-Seeking Loss: {risk_seeking_grad.item()}
            Total Loss: {loss.item()}
            Best Performance (Overall): {best_performance}
            Best Expression (Overall): {best_expression}
            Best Performance (Epoch): {max(rewards)}
            Best Expression (Epoch): {best_epoch_expression}
            """)
        if Pareto_frontier:
            print("Pareto frontier in the current batch:")
            print("")
            print("Complexity #  {} Loss #  Expression".format(reward_type))
            for pareto_i in range(len(PA.get_pareto_points())):
                print(np.round(PA.get_pareto_points()[pareto_i][0], 2), np.round(PA.get_pareto_points()[pareto_i][1], 2),
                      PA.get_pareto_points()[pareto_i][2])
            print("")

        # Sample for next batch
        # sequences, sequence_lengths, log_probabilities, entropies = dsr_rnn.sample_sequence(batch_size)
        sequences, sequence_lengths, log_probabilities, entropies = dsr_rnn.sample_n_expressions(batch_size)

    if summary_print:
        print(f"""
        Time Elapsed: {round(float(time.time() - start), 2)}s
        Epochs Required: {i + 1}
        Total mean length: {np.mean(epoch_mean_length)}
        Best Performance: {round(best_performance.item(), 3)}
        Best Expression: {best_expression}
        """)

    return [epoch_best_rewards, epoch_best_expressions, best_performance, best_expression, epoch_mean_rewards]