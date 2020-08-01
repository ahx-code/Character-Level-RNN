from argparse import ArgumentParser


def arguments(learning_rate=0.005, train_size=100000,
              test_size=10000, log_interval=5000,
              plot_every=1000, save_model=True, seed=1):
    """
    Args:
        learning_rate (float, optional): Step size of each
            iteration
        train_size (int, optional): Training-set size
        test_size (int, optional): Test-set size
        log_interval (int, optional): Step to save batches
        plot_every (int, optional): Saves each
            current_loss / args.plot_every to the loss list
        save_model (bool, optional): If true saves rnn model
        seed (int, optional): Value for producing
            the same results.

    Returns:
        Namespace: argument object
    """
    parser = ArgumentParser(description="NLP Example-1")
    parser.add_argument('--lr', type=float,
                        default=learning_rate,
                        metavar='LR',
                        help='learning rate'
                             'default: {}'.
                        format(learning_rate))
    parser.add_argument('--train-size', type=int,
                        default=train_size,
                        metavar='IN',
                        help='iteration number'
                             'default: {}'
                        .format(train_size))
    parser.add_argument('--test-size', type=int,
                        default=test_size,
                        help='Item number to be '
                             'displayed in the '
                             'confusion matrix. '
                             'default: {}'
                        .format(test_size))
    parser.add_argument('--log-interval', type=int,
                        default=log_interval,
                        help='Prints every {} prediction'
                        .format(log_interval))
    parser.add_argument('--plot-every', type=int,
                        default=plot_every,
                        help='Add each '
                             '{} to the loss'
                        .format(plot_every))
    parser.add_argument('--hidden-size', type=int)
    parser.add_argument('--save-model',
                        action='store_true',
                        default=save_model,
                        help='For Saving the current Model '
                             '(default: {})'
                        .format(save_model))
    parser.add_argument('--seed', type=int, default=seed,
                        metavar='S',
                        help='random seed (default: {})'.
                        format(seed))
    parser.add_argument('--num-layers', type=int)
    parser.add_argument('--dropout', type=float,
                        help='Drop the neurons with '
                             'the given probability ')
    parser.add_argument('--rnn-type', type=str,
                        help='rnn type can be: RNN, LSTM,'
                             'or GRU')
    argument_object = parser.parse_args()
    return argument_object
