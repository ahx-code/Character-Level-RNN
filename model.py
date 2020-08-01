from torch import cat, zeros
from torch import nn
from torch.nn import Module, Linear
from torch.nn import Dropout, LogSoftmax, Embedding
from torch.nn.functional import relu, log_softmax
from torch.nn.init import uniform_, zeros_


class RNNTutorial(Module):
    """`RNN model tutorial`_

    .. _RNN model tutorial:
        https://github.com/pytorch/examples/
        blob/master/word_language_model/model.py
    """

    def __init__(self, input_size, hidden_size,
                 output_size):
        super(RNNTutorial, self).__init__()
        self.hidden_size = hidden_size
        size_sum = input_size + hidden_size
        self.i2h = Linear(size_sum, hidden_size)
        self.i2o = Linear(size_sum, output_size)
        self.softmax = LogSoftmax(dim=1)

    def forward(self, input_, hidden_):
        combined = cat(tensors=(input_, hidden_), dim=1)
        hidden_ = self.i2h(input=combined)
        hidden_ = relu(hidden_)
        output = self.i2o(input=combined)
        output = self.softmax(input=output)
        return output, hidden_

    def init_hidden(self):
        return zeros(1, self.hidden_size)


class RNNModel(Module):
    """Inspired from the `PyTorch LSTM model`_

    .. _PyTorch LSTM model:
        https://github.com/pytorch/examples/
        blob/master/word_language_model/model.py
    """
    def __init__(self, rnn_type, input_size,
                 hidden_size, num_layers, dropout,
                 output_size):
        super(RNNModel, self).__init__()
        self.token_size = output_size
        self.drop = Dropout(p=dropout)
        self.mul_size = input_size * hidden_size
        self.rnn_type = rnn_type

        self.encoder = Embedding(num_embeddings=output_size,
                                 embedding_dim=input_size)

        self.rnn = getattr(nn, rnn_type)(input_size,
                                         hidden_size,
                                         num_layers,
                                         dropout=dropout)

        self.decoder = Linear(in_features=self.mul_size,
                              out_features=output_size)
        self.init_weights()
        self.hidden_size = hidden_size
        self.weight_size = (num_layers, input_size,
                            hidden_size)

    def init_weights(self):
        uniform_(tensor=self.encoder.weight, a=-0.1, b=0.1)
        zeros_(tensor=self.decoder.weight)
        uniform_(tensor=self.decoder.weight, a=-0.1, b=0.1)

    def forward(self, input_, hidden_):
        embedded = self.drop(self.encoder(input_))
        output, hidden_ = self.rnn(embedded, hidden_)
        output = self.drop(output)
        output = output.reshape((1, self.mul_size))
        decoded = self.decoder(output)
        softmax = log_softmax(input=decoded, dim=1)
        return softmax, hidden_

    def init_hidden(self):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.weight_size),
                    weight.new_zeros(self.weight_size))
        elif self.rnn_type == 'GRU':
            return weight.new_zeros(self.weight_size)
        else:
            raise Exception('rnn type can be either '
                            'LSTM or GRU, current '
                            'rnn type is {}'.
                            format(self.rnn_type))
