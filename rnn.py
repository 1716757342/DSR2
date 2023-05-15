###############################################################################
# General Information
###############################################################################
# Author: Daniel DiPietro | dandipietro.com | https://github.com/dandip

# Original Paper: https://arxiv.org/abs/1912.04871 (Petersen et al)

# rnn.py: Houses the RNN model used to sample expressions. Supports batched
# sampling of variable length sequences. Can select RNN, LSTM, or GRU models.

###############################################################################
# Dependencies
###############################################################################
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from prior import LengthConstraint
from dnc import DNC
###############################################################################
# Sequence RNN Class
###############################################################################
from subroutines import parents_siblings


class DSRRNN(nn.Module):
    OBS_DIM = 4

    def __init__(self, operators, hidden_size, device, min_length=4, max_length=30, type='rnn', num_layers=1, dropout=0.0, prior=None,
                 entropy_gamma=None):
        super(DSRRNN, self).__init__()

        # self.input_size = 2 * len(operators)  # One-hot encoded parent and sibling
        self.input_size = operators.n_parent_inputs + operators.n_sibling_inputs  # for sample_n_expressions()
        self.hidden_size = hidden_size
        self.output_size = len(operators)  # Output is a softmax distribution over all operators
        self.num_layers = num_layers
        self.dropout = dropout
        self.operators = operators
        self.device = device
        self.type = type

        self.prior = prior
        self.entropy_gamma = entropy_gamma

        # Initial cell optimization
        self.init_input = nn.Parameter(data=torch.rand(1, self.input_size), requires_grad=True).to(self.device)
        self.init_hidden = nn.Parameter(data=torch.rand(self.num_layers, self.hidden_size), requires_grad=True).to(self.device)

        self.min_length = min_length
        self.max_length = max_length

        # Find max_length from the LengthConstraint prior, if it exists
        # Both priors will never happen in the same experiment
        prior_max_length = None
        for single_prior in self.prior.priors:
            if isinstance(single_prior, LengthConstraint):
                if single_prior.max is not None:
                    prior_max_length = single_prior.max
                    self.max_length = prior_max_length
                break

        if prior_max_length is None:
            assert max_length is not None, "max_length must be specified if " \
                                           "there is no LengthConstraint."
            self.max_length = max_length
            print("WARNING: Maximum length not constrained. Sequences will "
                  "stop at {} and complete by repeating the first input "
                  "variable.".format(self.max_length))
        elif max_length is not None and max_length != self.max_length:
            print("WARNING: max_length ({}) will be overridden by value from "
                  "LengthConstraint ({}).".format(max_length, self.max_length))

        max_length = self.max_length

        # Entropy decay vector
        if self.entropy_gamma is None:
            self.entropy_gamma = 1.0
        self.entropy_gamma_decay = torch.tensor([self.entropy_gamma ** t for t in range(max_length)])

        if self.type == 'rnn':
            self.rnn = nn.RNN(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout
            )
            self.projection_layer = nn.Linear(self.hidden_size, self.output_size).to(self.device)
        elif self.type == 'lstm':
            self.lstm = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                proj_size=self.output_size,
                dropout=self.dropout
            ).to(self.device)
            self.init_hidden_lstm = nn.Parameter(data=torch.rand(self.num_layers, self.output_size), requires_grad=True).to(self.device)
        elif self.type == 'gru':
            self.gru = nn.GRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout
            )
            self.projection_layer = nn.Linear(self.hidden_size, self.output_size).to(self.device)
        elif self.type == 'dnc':
            self.dnc = DNC(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                rnn_type='lstm',
                num_layers=self.num_layers,
                nr_cells=5,
                cell_size=10,
                read_heads=2,
                batch_first=True,
                gpu_id=-1
            )
            self.projection_layer = nn.Linear(self.input_size, self.output_size).to(self.device)
        self.activation = nn.Softmax(dim=1)

    # def sample_sequence(self, n):
    #     # 同时采样n个表达式
    #     sequences = torch.zeros((n, 0))
    #     entropies = torch.zeros((n, 0))  # Entropy for each sequence
    #     log_probs = torch.zeros((n, 0))  # Log probability for each token
    #
    #     sequence_mask = torch.ones((n, 1), dtype=torch.bool)
    #
    #     input_tensor = self.init_input.repeat(n, 1)  # [n, 2*ops]
    #     hidden_tensor = self.init_hidden.repeat(n, 1)  # [n, hid_size]
    #     if self.type == 'lstm':
    #         hidden_lstm = self.init_hidden_lstm.repeat(n, 1)  # [n, ops]
    #
    #     counters = torch.ones(n)  # 完成表达式采样还需要的节点数
    #     lengths = torch.zeros(n)  # 当前表达式的长度
    #
    #     # While there are still tokens left for sequences in the batch只要有一个式子没采完就继续采
    #     while sequence_mask.all(dim=1).any():
    #         if self.type == 'rnn':
    #             output, hidden_tensor = self.forward(input_tensor, hidden_tensor)
    #         elif self.type == 'lstm':
    #             output, hidden_tensor, hidden_lstm = self.forward(input_tensor, hidden_tensor, hidden_lstm)  # output[n, lib_size]
    #         elif self.type == 'gru':
    #             output, hidden_tensor = self.forward(input_tensor, hidden_tensor)
    #         elif self.type == 'dnc':
    #             (controller_hidden, memory, read_vectors) = (None, None, None)
    #             output, (controller_hidden, memory, read_vectors) = self.forward(input_tensor, (controller_hidden, memory, read_vectors), reset_experience=True)
    #
    #         # Apply constraints and normalize distribution
    #         output = self.apply_constraints(output, counters, lengths, sequences)
    #         output = output / torch.sum(output, axis=1)[:, None]
    #
    #         # Sample from categorical distribution
    #         dist = torch.distributions.Categorical(output)  # 根据类别概率分布进行采样,概率大的被采到的几率大
    #         token = dist.sample()  # [n]
    #
    #         # Add sampled tokens to sequences
    #         sequences = torch.cat((sequences, token[:, None]), axis=1)
    #         lengths += 1
    #
    #         # Add log probability of current token
    #         log_probs = torch.cat((log_probs, dist.log_prob(token)[:, None]), axis=1)  # dist.log_prob计算对应token的ln值
    #
    #         # Add entropy of current token
    #         entropies = torch.cat((entropies, dist.entropy()[:, None]), axis=1)  # 计算token的熵值, 每个token：概率分布分别计算 -p*lnp
    #
    #         # Update counter
    #         counters -= 1
    #         counters += torch.isin(token, self.operators.arity_two).long() * 2  # 判断token中的元素是否在arity中，在返回True不在返回False, 返回的形状和token相同
    #         counters += torch.isin(token, self.operators.arity_one).long() * 1
    #
    #         # Update sequence mask
    #         # This is for the next token that we sample. Basically, we know if the
    #         # next token will be valid or not based on whether we've just completed the sequence (or have in the past)
    #         sequence_mask = torch.cat(
    #             (sequence_mask, torch.bitwise_and((counters > 0)[:, None], sequence_mask.all(dim=1)[:, None])),  # 输入是布尔类型计算逻辑与
    #             axis=1)
    #
    #         # Compute next parent and sibling; assemble next input tensor
    #         parent_sibling = self.get_parent_sibling(sequences, lengths)
    #         input_tensor = self.get_next_input(parent_sibling)
    #
    #     # Filter entropies log probabilities using the sequence_mask
    #     entropies = torch.sum(entropies * (sequence_mask[:, :-1]).long(), axis=1)
    #     log_probs = torch.sum(log_probs * (sequence_mask[:, :-1]).long(), axis=1)
    #     sequence_lengths = torch.sum(sequence_mask.long(), axis=1)
    #
    #     return sequences, sequence_lengths, log_probs, entropies

    def sample_n_expressions(self, n):
        """将tensorflow版本的采样和约束添加过程移植进来"""

        sequences = torch.zeros((n, 0))
        entropies = torch.zeros((n, 0))  # Entropy for each sequence
        log_probs = torch.zeros((n, 0))  # Log probability for each token

        sequence_mask = torch.ones((n, 1), dtype=torch.bool)

        counters = torch.ones(n)  # 完成表达式采样还需要的节点数
        lengths = torch.zeros(n)  # 当前表达式的长度

        # Order of observations: action, parent, sibling, dangling
        initial_obs = torch.tensor([self.operators.EMPTY_ACTION,
                                    self.operators.EMPTY_PARENT,
                                    self.operators.EMPTY_SIBLING,
                                    1], dtype=torch.float32)
        initial_obs = initial_obs.repeat(n, 1)  # [batch_size, obs_dim]
        obs = initial_obs
        initial_prior = torch.from_numpy(self.prior.initial_prior())
        initial_prior = initial_prior.repeat(n, 1)  # [batch_size, n_choices]
        prior = initial_prior

        input_tensor = self.get_tensor_input(initial_obs)  # [batch_size, n_parent_inputs + n_sibling_inputs]

        hidden_tensor = self.init_hidden.repeat(n, 1)  # [batch_size, hid_dim]
        if self.type == 'lstm':
            hidden_lstm = self.init_hidden_lstm.repeat(n, 1)  # [batch_size, n_choices]

        while sequence_mask.all(dim=1).any():
            if self.type == 'rnn':
                output, hidden_tensor = self.forward(input_tensor, hidden_tensor)
            elif self.type == 'lstm':
                output, hidden_tensor, hidden_lstm = self.forward(input_tensor, hidden_tensor, hidden_lstm)  # output[n, lib_size]
            elif self.type == 'gru':
                output, hidden_tensor = self.forward(input_tensor, hidden_tensor)
            elif self.type == 'dnc':
                (controller_hidden, memory, read_vectors) = (None, None, None)
                output, (controller_hidden, memory, read_vectors) = self.forward(input_tensor, (controller_hidden, memory, read_vectors),
                                                                                 reset_experience=True)

            output = output + prior  # 将违反约束的token概率置-inf
            output = torch.where(torch.isinf(output), torch.full_like(output, 0), output)  # 将-inf替换为0
            output = output / torch.sum(output, axis=1)[:, None]  # 概率重新归一化

            dist = torch.distributions.Categorical(output)
            token = dist.sample()  # [batch_size]

            sequences = torch.cat((sequences, token[:, None]), axis=1)
            lengths += 1

            # Add log probability of current token
            log_probs = torch.cat((log_probs, dist.log_prob(token)[:, None]), axis=1)  # dist.log_prob计算对应token的ln值

            # Add entropy of current token
            entropies = torch.cat((entropies, dist.entropy()[:, None]), axis=1)  # 计算token的熵值, 每个token：概率分布分别计算 -p*lnp

            # Update counter
            counters -= 1
            counters += torch.isin(token, self.operators.arity_two).long() * 2  # 判断token中的元素是否在arity中，在返回True不在返回False, 返回的形状和token相同
            counters += torch.isin(token, self.operators.arity_one).long() * 1

            sequence_mask = torch.cat((sequence_mask, torch.bitwise_and((counters > 0)[:, None], sequence_mask.all(dim=1)[:, None])),
                                      axis=1)  # 输入是布尔类型计算逻辑与

            # Compute next parent and sibling; assemble next input tensor
            next_obs, next_prior = self.get_next_obs(sequences, obs)
            next_obs, next_prior = torch.from_numpy(next_obs), torch.from_numpy(next_prior)

            input_tensor = self.get_tensor_input(next_obs)
            prior = next_prior
            obs = next_obs

        # Filter entropies log probabilities using the sequence_mask
        # 新添加熵的decay
        # self.entropy_gamma_decay_mask = self.entropy_gamma_decay[:len(sequence_mask)] * (sequence_mask[:, :-1].long())
        # entropies = torch.sum(entropies * self.entropy_gamma_decay_mask, axis=1)
        entropies = torch.sum(entropies * (sequence_mask[:, :-1]).long(), axis=1)  # 完整表达式的熵
        log_probs = torch.sum(log_probs * (sequence_mask[:, :-1]).long(), axis=1)
        sequence_lengths = torch.sum(sequence_mask.long(), axis=1)
        return sequences, sequence_lengths, log_probs, entropies

    def forward(self, input, hidden, hidden_lstm=None, reset_experience=None):
        """Input should be [parent, sibling]
        """
        if self.type == 'rnn':
            output, hidden = self.rnn(input[:, None].float(), hidden[None, :])
            output = output[:, 0, :]
            output = self.projection_layer(output)
            output = self.activation(output)
            return output, hidden[0, :]
        elif self.type == 'lstm':
            output, (hn, cn) = self.lstm(input[:, None].float(), (hidden_lstm[None, :], hidden[None, :]))  # output[batch_size, len, hid_dim]
            output = self.activation(output[:, 0, :])  # [batch_size, n_choices]
            return output, cn[0, :], hn[0, :]
        elif self.type == 'gru':
            output, hn = self.gru(input[:, None].float(), hidden[None, :])
            output = output[:, 0, :]
            output = self.projection_layer(output)
            output = self.activation(output)
            return output, hn[0, :]
        elif self.type == 'dnc':
            output, (controller_hidden, memory, read_vectors) = self.dnc(input[:, None].float(), hidden,
                                                                         reset_experience=True)  # output[batch_size, 1, input_size]
            output = output[:, 0, :]
            output = self.projection_layer(output)
            output = self.activation(output)
            return output, (controller_hidden, memory, read_vectors)

    def apply_constraints(self, output, counters, lengths, sequences):
        """Applies in situ constraints to the distribution contained in output based on the current tokens
        """
        # Add small epsilon to output so that there is a probability of selecting
        # everything. Otherwise, constraints may make the only operators ones
        # that were initially set to zero, which will prevent us selecting
        # anything, resulting in an error being thrown
        epsilon = torch.ones(output.shape) * 1e-20
        output = output + epsilon.to(self.device)

        # ~ 检查表达式长度不能小于最小长度 ~
        # Explanation here
        min_boolean_mask = (counters + lengths >= torch.ones(counters.shape) * self.min_length).long()[:, None]  # 判断每个式子有没有比最小长度大，大于的话值为1，不够的值为0
        min_length_mask = torch.max(self.operators.nonzero_arity_mask[None, :], min_boolean_mask)
        output = torch.minimum(output, min_length_mask)  # 返回两个tensor成对比较的最小值 -> 将x符号的概率置0, 因为没有达到最小长度，所以不能采x。比较的关键就是non_zero_arity_mask中x的位置为0

        # ~ 检查表达式长度不能超过最大长度 ~
        max_boolean_mask = (counters + lengths <= torch.ones(counters.shape) * (self.max_length - 2)).long()[:,
                           None]  # 判断每个式子是不是小于最大长度-2，小于的话值为1，超了的为0
        max_length_mask = torch.max(self.operators.zero_arity_mask[None, :],
                                    max_boolean_mask)  # 如果超过最大长度-2，利用zero_arity_mask将最后x的概率置1其他全置0，强行采x填充剩下的两个位置完成表达式采样
        output = torch.minimum(output, max_length_mask)

        # ~ 确保所有表达式都有一个变量 ~
        nonvar_zeroarity_mask = (
            ~torch.logical_and(self.operators.zero_arity_mask, self.operators.nonvariable_mask)).long()  # 元素之间进行逻辑与运算，0为False1为True
        if lengths[0].item() == 0.0:  # First thing we sample can't be
            output = torch.minimum(output, nonvar_zeroarity_mask)
        else:
            nonvar_zeroarity_mask = nonvar_zeroarity_mask.repeat(counters.shape[0], 1)
            # Don't sample a nonvar zeroarity token if the counter is at 1 and
            # we haven't sampled a variable yet
            counter_mask = (counters == 1)
            contains_novar_mask = ~(torch.isin(sequences, self.operators.variable_tensor).any(axis=1))
            last_token_and_no_var_mask = (~torch.logical_and(counter_mask, contains_novar_mask)[:, None]).long()
            nonvar_zeroarity_mask = torch.max(nonvar_zeroarity_mask, last_token_and_no_var_mask * torch.ones(nonvar_zeroarity_mask.shape)).long()
            output = torch.minimum(output, nonvar_zeroarity_mask)

            # ~ 加入三角函数和对指数不能嵌套的约束～
            parent = self.get_parent_sibling(sequences, lengths)[:, 0]  # [n] 取出当前parent
            have_sin_cos = torch.isin(parent, self.operators.cos_sin_index)[:, None].long()  # [n, 1] parent是三角函数的表达式为True，然后变为1
            nest_sin_cos_mask = torch.logical_and(self.operators.non_sin_cos_mask, have_sin_cos)  # [n, lib_size] 对应要概率置0的mask保持，其余表达式的mask为False
            nest_sin_cos_mask = nest_sin_cos_mask.long() + (~have_sin_cos.bool()).long()  # [n, lib_size] 对应概率要置0的表达式保持mask, 其余全为1
            output = torch.minimum(output, nest_sin_cos_mask)  # 将对应表达式对应的sin cos概率置0

            have_exp_ln = torch.isin(parent, self.operators.exp_ln_index)[:, None].long()  # [n, 1] parent是对指数的表达式为True，然后变为1
            nest_exp_ln_mask = torch.logical_and(self.operators.non_exp_ln_mask, have_exp_ln)  # [n, lib_size] 对应要概率置0的mask保持，其余表达式的mask为0
            nest_exp_ln_mask = nest_exp_ln_mask.long() + (~have_exp_ln.bool()).long()  # [n, lib_size] 对应概率要置0的表达式保持mask, 其余全为1
            output = torch.minimum(output, nest_exp_ln_mask)  # 将对应表达式对应的log exp概率置0

        return output

    def get_parent_sibling(self, sequences, lengths):
        """Returns parent, sibling for the most recent token in token_list
        """
        parent_sibling = torch.ones((lengths.shape[0], 2)) * -1
        recent = int(lengths[0].item()) - 1

        c = torch.zeros(lengths.shape[0])
        for i in range(recent, -1, -1):
            # Determine arity of the i-th tokens
            token_i = sequences[:, i]
            arity = torch.zeros(lengths.shape[0])
            arity += torch.isin(token_i, self.operators.arity_two).long() * 2
            arity += torch.isin(token_i, self.operators.arity_one).long() * 1

            # Increment c by arity of the i-th token, minus 1
            c += arity
            c -= 1

            # In locations where c is zero (and parents and siblings that are -1),
            # we want to set parent_sibling to sequences[:, i] and sequeneces[:, i+1].
            # c_mask an n x 1 tensor that is True when c is zero and parents/siblings are -1.
            # It is False otherwise.
            c_mask = torch.logical_and(c == 0, (parent_sibling == -1).all(axis=1))[:, None]

            # n x 2 tensor where dimension is 2 is sequences[:, i:i+1]
            # Since i+1 won't exist on the first iteration, we pad
            # (-1 is i+1 doesn't exist)
            i_ip1 = sequences[:, i:i + 2]
            i_ip1 = F.pad(i_ip1, (0, 1), value=-1)[:, 0:2]  # (0, 1)表示对矩阵最后一维进行填充，左边不填充，右边填充1，填充的值为-1

            # Set i_ip1 to 0 for indices where c_mask is False
            i_ip1 = i_ip1 * c_mask.long()

            # Set parent_sibling to 0 for indices where c_mask is True
            parent_sibling = parent_sibling * (~c_mask).long()

            parent_sibling = parent_sibling + i_ip1

        ###
        # If our most recent token has non-zero arity, then it is the
        # parent, and there is no sibling. We use the following procedure:
        ###

        # We create recent_nonzero_mask, an n x 1 tensor that is True if the most
        # recent token has non-zero arity and False otherwise.
        recent_nonzero_mask = (~torch.isin(sequences[:, recent], self.operators.arity_zero))[:, None]

        # This sets parent_sibling to 0 for instances where recent_nonzero_mask is True
        # If recent_nonzero_mask is False, the value of parent_sibling is unchanged
        parent_sibling = parent_sibling * (~recent_nonzero_mask).long()

        # This tensor is n x 2 where the 2 dimension is [recent token, -1]
        recent_parent_sibling = torch.cat((sequences[:, recent, None], -1 * torch.ones((lengths.shape[0], 1))), axis=1)

        # We set values of recent_parent_sibling where recent_nonzero_mask is False
        # to zero.
        recent_parent_sibling = recent_parent_sibling * recent_nonzero_mask.long()

        # Finally, add recent_parent_sibling to parent_sibling.
        parent_sibling = parent_sibling + recent_parent_sibling

        return parent_sibling

    def get_next_input(self, parent_sibling):
        # Just convert -1 to 1 for now; it'll be zeroed out later
        parent = torch.abs(parent_sibling[:, 0]).long()
        sibling = torch.abs(parent_sibling[:, 1]).long()

        # Generate one-hot encoded tensors
        parent_onehot = F.one_hot(parent, num_classes=len(self.operators))
        sibling_onehot = F.one_hot(sibling, num_classes=len(self.operators))

        # Use a mask to zero out values that are -1. Parent should never be -1,
        # but we do it anyway.
        parent_mask = (~(parent_sibling[:, 0] == -1)).long()[:, None]
        parent_onehot = parent_onehot * parent_mask
        sibling_mask = (~(parent_sibling[:, 1] == -1)).long()[:, None]
        sibling_onehot = sibling_onehot * sibling_mask

        input_tensor = torch.cat((parent_onehot, sibling_onehot), axis=1)
        return input_tensor

    def get_tensor_input(self, obs):
        """获取RNN的输入：parent||sibling, obs: [action, parent, sibling, dangling]"""
        parent = torch.abs(obs[:, 1]).long()  # [batch_size]
        sibling = torch.abs(obs[:, 2]).long()  # [batch_size]

        parent_onehot = F.one_hot(parent, num_classes=self.operators.n_parent_inputs)
        sibling_onehot = F.one_hot(sibling, num_classes=self.operators.n_sibling_inputs)

        # Use a mask to zero out values that are -1. Parent should never be -1,
        # but we do it anyway.
        parent_mask = (~(obs[:, 1] == -1)).long()[:, None]
        parent_onehot = parent_onehot * parent_mask
        sibling_mask = (~(obs[:, 1] == -1)).long()[:, None]
        sibling_onehot = sibling_onehot * sibling_mask

        input_tensor = torch.cat([parent_onehot, sibling_onehot], dim=1)
        return input_tensor

    def get_next_obs(self, actions, obs):
        # 为了get_next_obs使用numba加速，需要先把tensor转换成ndarray
        actions = actions.numpy().astype(np.int32)
        obs = obs.numpy().astype(np.int32)

        dangling = obs[:, 3]  # Shape of obs: (?, 4)
        action = actions[:, -1]  # Current action
        lib = self.operators

        # Compute parents and siblings
        parent, sibling = parents_siblings(actions,
                                           arities=lib.arities,
                                           parent_adjust=lib.parent_adjust,
                                           empty_parent=lib.EMPTY_PARENT,
                                           empty_sibling=lib.EMPTY_SIBLING)

        # Update dangling with (arity - 1) for each element in action
        dangling += lib.arities[action] - 1

        prior = self.prior(actions, parent, sibling, dangling)  # (batch_size, n_choices)

        next_obs = np.stack([action, parent, sibling, dangling], axis=1)  # (?, 4)
        next_obs = next_obs.astype(np.float32)
        return next_obs, prior
