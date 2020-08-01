from torch import load, no_grad, zeros, save
from torch import Tensor, tensor, long
from torch.nn import Module, NLLLoss
from sklearn.metrics import accuracy_score
from util import line2tensor
from time import time
from util import category_from_output
from util import time_since, random_choice
from graphs import plot_loss, plot_confusion
from os.path import join


def example_data(all_cat, cat_line, rnn_type):
    """Display category / line for 10 random samples

    Args:
        all_cat (list): all categories in dataset
        cat_line (dict): category lines
        rnn_type (str): Either RNN or LSTM
    """
    cat_ = random_choice(category_=all_cat)
    line = random_choice(category_=cat_line[cat_])
    temporary = [all_cat.index(cat_)]
    cat_tensor = tensor(data=temporary, dtype=long)
    line_tensor = line2tensor(line, rnn_type)

    return cat_, line, cat_tensor, line_tensor


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history. `source`_.

    .. _source:
        https://github.com/floydhub/
        word-language-model/blob/master/main.py
    """

    if isinstance(h, Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def train(cat_tensor, line_tensor, rnn,
          criterion, lr, rnn_type, output_size):
    """Train the RNN

    Args:
        cat_tensor (Tensor): category tensor
        line_tensor (Tensor): output of the
            line2tensor method
        rnn (Module): RNN object
        criterion (NLLLoss): Useful to train a
            classification problem with `C` classes.
        lr (float): learning-rate
        rnn_type (str): Either RNN or LSTM
        output_size (int):

    Returns:
        tuple: (output, loss.item())
    """
    if not rnn_type == 'RNN':
        rnn.train()

    hidden = rnn.init_hidden()
    rnn.zero_grad()
    output = zeros(size=(1, output_size))
    for i in range(line_tensor.size()[0]):
        if rnn_type == 'LSTM' or rnn_type == 'GRU':
            hidden = repackage_hidden(hidden)
        else:
            output, hidden = rnn(input_=line_tensor[i],
                                 hidden_=hidden)
    loss = criterion(input=output, target=cat_tensor)
    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-lr)

    return output, loss.item()


def evaluate(rnn, line_tensor, output_size):
    hidden = rnn.init_hidden()
    output = zeros(size=(1, output_size))

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(input_=line_tensor[i],
                             hidden_=hidden)

    return output


def predict(input_line, rnn, all_cat, n_pred, rnn_type):
    print('\n> %s' % input_line)
    with no_grad():
        line_tensor = line2tensor(input_line, rnn_type)
        output_size = len(all_cat)
        out = evaluate(rnn, line_tensor, output_size)
        topv, topi = out.topk(n_pred, dim=1, largest=True)
        for i in range(n_pred):
            val = topv[0][i].item()
            cat_idx = topi[0][i].item()
            print('(%.2f) %s' % (val, all_cat[cat_idx]))


def train_rnn(arg_, all_cat, cat_line, rnn_object,
              folder, rnn_type, output_size, model_name):
    """Train the recursive neural network object

    Args:
        arg_ (Namespace): argument object
        all_cat (list): all categories
        cat_line (dict): each corresponding category line
        rnn_object (RNN): rnn module object
        folder (str): folder name,
        rnn_type (str): Either RNN or LSTM
        output_size (int): network output length
        model_name (str): Name of the model to be saved.
    """
    print("\nTraining network for {} samples..".
          format(arg_.train_size))
    criterion = NLLLoss()
    current_loss = 0
    all_losses = []
    start = time()

    for _iter in range(1, arg_.train_size + 1):
        _cat, _line, cat_tensor, line_tensor = \
            example_data(all_cat, cat_line, rnn_type)

        output, loss = train(cat_tensor, line_tensor,
                             rnn_object, criterion,
                             arg_.lr, rnn_type,
                             output_size)

        current_loss += loss

        if _iter % arg_.log_interval == 0:
            guess, guess_i = \
                category_from_output(output, all_cat)
            if guess == _cat:
                correct = '✓'
            else:
                correct = '✗ (%s)' % _cat

            print('%d %d%% (%s) %.4f %s / %s %s' % (
                _iter, _iter / arg_.train_size * 100,
                time_since(start),
                loss, "".join(_line), guess, correct))

            if _iter % arg_.log_interval == 0:
                div = current_loss / arg_.plot_every
                all_losses.append(div)
                current_loss = 0

    plot_loss(train_loss=all_losses, folder=folder)

    if arg_.save_model:
        file_save = join(folder, model_name)
        save(obj=rnn_object.state_dict(), f=file_save)


def evaluate_rnn(arg_, all_cat, cat_line, rnn_object,
                 folder, rnn_type, model_name):
    """Evaluates the recursive neural network
        object performance

    Args:
        arg_ (Namespace): argument object
        all_cat (list): all categories
        cat_line (dict): each corresponding category line
        rnn_object (RNN): rnn module object
        folder (str): folder name,
        rnn_type (str): Either RNN or LSTM
        model_name (str): Name of the model to be loaded.
    """
    y_true = []
    y_pred = []

    print("\nLoading {}...".format(model_name))
    state_dict = join(folder, model_name)
    rnn_object.load_state_dict(state_dict=load(state_dict))

    for i in range(arg_.test_size):
        _cat, _line, cat_tensor, line_tensor = \
            example_data(all_cat, cat_line, rnn_type)

        output = evaluate(rnn_object, line_tensor,
                          output_size=len(all_cat))

        guess, _ = category_from_output(output,
                                        all_cat=all_cat)
        y_true.append(_cat)
        y_pred.append(guess)

    accuracy_val = accuracy_score(y_true, y_pred) * 100
    print("\nAccuracy : {:,.2f}%".format(accuracy_val))

    # Graphs
    plot_confusion(y_true, y_pred, all_cat,
                   norm='true', path=folder)
    plot_confusion(y_true, y_pred, all_cat,
                   norm=None, path=folder)

    n_pred = 3  # number of the prediction

    predict('Dovesky', rnn_object, all_cat,
            n_pred, rnn_type)
    predict('Jackson', rnn_object, all_cat,
            n_pred, rnn_type)
    predict('Satoshi', rnn_object, all_cat,
            n_pred, rnn_type)
