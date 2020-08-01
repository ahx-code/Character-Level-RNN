from methods import example_data
from letters import AllLetters
import util as m


def examples(all_cat, cat_line, n_hidden_,
             num_layers, rnn, rnn_type):
    n_letters = AllLetters.n_letters
    ex_name = 'Ślusàrski'
    ascii_ex_name = "".join(m.unicode2ascii(s=ex_name))
    lang = "".join(cat_line['Italian'][0])
    print("\nSome examples:")
    print("\n{} -> {}".format(ex_name, ascii_ex_name))
    print("\nSome Italian characters: {}".format(lang))
    line = 'J'
    print("\nJ tensor representation -> {}".
          format(m.line2tensor(line, rnn_type)))
    print("\nJ size:\t\t {}".
          format(m.line2tensor('J', rnn_type).size()))
    line = 'Jones'
    print("Jones size:  {}".
          format(m.line2tensor(line, rnn_type).size()))
    line = 'Albert'
    input_ = m.line2tensor(line, rnn_type)

    if rnn_type == 'LSTM' or rnn_type == 'GRU':
        size = (num_layers, n_letters, n_hidden_)
        if rnn_type == 'LSTM':
            hidden_ = (m.zeros(size), m.zeros(size))
        else:
            hidden_ = m.zeros(size)
        output_, hidden_ = rnn(input_[0], hidden_)
    elif rnn_type == 'RNN':
        hidden_ = m.zeros(size=(1, n_hidden_))
        output_, next_hidden = rnn(input_[0], hidden_)
    else:
        raise Exception('rnn_type is not supported.')

    predicted_name = m.category_from_output(output_,
                                            all_cat)[0]

    print("\nExample word: {} is predicted as {} word".
          format(line, predicted_name))
    print("\nSome other tuples (category/line):")

    for i in range(10):
        _cat, line, _, _ = example_data(all_cat, cat_line,
                                        rnn_type)
        print('{} -> {}'.format(_cat, "".join(line)))
