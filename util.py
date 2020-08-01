from glob import glob
from random import randint
from os.path import splitext, basename
from unicodedata import normalize, category
from torch import zeros
from shutil import rmtree
from os import mkdir
from os.path import exists, join
from letters import AllLetters
from math import floor
from time import time
from torch import int64


def unicode2ascii(s):
    """Turns unicode string to plain ASCII using NFD.

    NFD (Normalization Form canonical Decomposition):
    ----
    Characters are decomposed by canonical equivalence.

    It has been explained in `Unicode Equivalence`_.

    Algorithm:
    ----------
    * Init ascii_list to empty list

    * For each character (c) in a sentence,
      find c's category.

        Categories:

        * Lu - Letter, Uppercase
        * Mn - Mark, nonspacing
        * Ll - Letter, lowercase

    * If c != Mn and c is a letter
      then ascii_list <- c

    Args:
        s (str): sentence

    .. _Unicode Equivalence:
        https://en.wikipedia.org/wiki/Unicode_equivalence

    Returns:
        list: (ascii_list) ascii equivalent of the s
    """
    letters = AllLetters.letters
    ascii_list = []
    for c in normalize('NFD', s):
        cat_c = category(c)
        if cat_c != 'Mn' and c in letters:
            ascii_list.append(c)
    return ascii_list


def find_files(pathname):
    """Find files from the given path.

    Args:
        pathname (str): data path
    Returns:
        list: (glob_object) text files
    """
    return glob(pathname=pathname, recursive=False)


def read_lines(filename):
    """Convert sentence into the
    character list

    Args:
        filename (str): file path
    Returns:
        list: (line_list) ascii lines
    """
    line_list = []
    lines = open(file=filename, encoding='utf-8').\
        read().strip().split(sep='\n')
    for line in lines:
        converted_line = unicode2ascii(s=line)
        line_list.append(converted_line)
    return line_list


def dataset(pathname):
    """Design dataset by storing names list in corresponding category

    Dataset contains:
        * all_categories: Czech.txt, Portuguese.txt     ..
        * category Lines:

            * Czech <- ['a', 'n'], ['t', 'w', 'x', 'z'] ..
            * Portuguese <- ['f', 'e', 'r'], ['a', 'n'] ..

    Args:
        pathname (str): data path
    Returns:
        tuple: (all_categories, category lines)
    """
    all_cat = []
    cat_line = {}
    for filename in find_files(pathname=pathname):
        base = basename(p=filename)
        text = splitext(p=base)
        cat_ = text[0]  # remove .txt extension
        all_cat.append(cat_)
        lines = read_lines(filename=filename)
        cat_line[cat_] = lines
    return all_cat, cat_line


def letter2index(letter):
    """Turns letter to the corresponding index

    Indexes
    -------
    a <- 0

    b <- 1

    c <- 2

    ...

    z <- 25

    A <- 26

    B <- 27

    ...

    Args:
        letter (str): input character
    Returns:
        int: corresponding index
    """
    letters = AllLetters.letters
    return letters.find(letter)


def letter2tensor(letter):
    """Turns letter to the <1 x n_letters> tensor

    Args:
        letter (str): input character

    Returns:
        tensor: one-hot vector
    """
    n_letters = AllLetters.n_letters
    tensor_ = zeros(1, n_letters)
    idx = letter2index(letter=letter)
    tensor_[0][idx] = 1
    return tensor_


def line2tensor(line, rnn_type):
    """Turns a line into a one-hot vector
    """
    n_letters = AllLetters.n_letters

    tensor_size = len(line), 1, n_letters

    if rnn_type == 'RNN':
        tensor_ = zeros(tensor_size)
    else:
        tensor_ = zeros(tensor_size).type(dtype=int64)

    for count, letter in enumerate(line):
        idx = letter2index(letter)
        tensor_[count][0][idx] = 1
    return tensor_


def category_from_output(output, all_cat):
    """Convert one-hot-vector to label

    Finds the first largest value in the output, using `topk`_.

    .. _topk:
        https://pytorch.org/docs/master/generated/torch.topk.html

    Returns:
        tuple: (category, index) of the output
    """
    top_n, top_i = output.topk(1)
    cat_i = top_i[0].item()
    return all_cat[cat_i], cat_i


def random_choice(category_):
    """Selects a random integer between
        0 - category length
    """
    idx = randint(0, len(category_)-1)
    choice = category_[idx]
    return choice


def create_folder(model_name):
    """Create folder name based on the model name

    Args:
        model_name (str): Model save name

    Returns:
         (str): folder name - the extension removed
            model name
    """
    folder_name = model_name.replace('.h5', '')

    if not exists(join(folder_name, model_name)):
        if exists(folder_name):
            rmtree(folder_name)
        mkdir(folder_name)

    return folder_name


def time_since(since):
    now = time()
    s = now - since
    m = floor(s/60)
    s -= m*60
    return '%dn %ds' % (m, s)


def get_model_name(rnn_type, num_layers, hidden_size, dropout):
    model_name = '{}_layer{}_hidden_size{}_dropout_{}.h5'.\
        format(rnn_type, num_layers, hidden_size, dropout)
    return model_name


def print_rnn_type(rnn_type):
    if rnn_type == 'GRU':
        print("\n\nSelected RNN Type: GRU")
    elif rnn_type == 'LSTM':
        print("\n\nSelected RNN Type: LSTM")
