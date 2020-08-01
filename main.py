# -*- coding: utf-8 -*-
from os.path import join, exists
from torch import manual_seed, device
from util import dataset, create_folder
from util import get_model_name
from util import print_rnn_type
from argument import arguments
from letters import AllLetters
from example import examples
from methods import train_rnn
from methods import evaluate_rnn
from model import RNNModel
from model import RNNTutorial


def main():
    args = arguments()

    manual_seed(args.seed)

    model_name = get_model_name(args.rnn_type, args.num_layers,
                                args.hidden_size, args.dropout)

    folder_name = create_folder(model_name)

    path = join('data', 'names', '*.txt')

    # Get all categories and the corresponding lines
    all_cat, cat_line = dataset(pathname=path)

    input_size = AllLetters.n_letters

    output_size = len(all_cat)

    _device = device("cpu")

    if args.rnn_type == 'LSTM' or args.rnn_type == 'GRU':

        print_rnn_type(args.rnn_type)

        rnn = RNNModel(args.rnn_type, input_size,
                       args.hidden_size, args.num_layers,
                       args.dropout, output_size).to(_device)

    elif args.rnn_type == 'RNN':

        print_rnn_type(args.rnn_type)

        rnn = RNNTutorial(input_size, args.hidden_size,
                          output_size).to(_device)

    else:
        raise Exception('rnn type can be LSTM, GRU or RNN')

    examples(all_cat, cat_line, args.hidden_size,
             args.num_layers, rnn, args.rnn_type)

    model_exists = exists(join(folder_name, model_name))

    if not model_exists:
        train_rnn(args, all_cat, cat_line, rnn, folder_name,
                  args.rnn_type, output_size, model_name)

    evaluate_rnn(args, all_cat, cat_line, rnn, folder_name,
                 args.rnn_type, model_name)


if __name__ == '__main__':
    main()
