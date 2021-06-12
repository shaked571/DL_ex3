import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import BiLSTMVanila, BiLSTMChar, BiLSTMSubWords, BiLSTMConcat
from trainer import pad_collate
from vocab import TokenVocab, CharsVocab, SubWords
from DataFiles import TokenDataFile


def dump_test_file(test_prediction, test_file_path, seperator, output_name):
    res = []
    cur_i = 0
    with open(test_file_path) as f:
        lines = f.readlines()
    for line in lines:
        if line == "" or line == "\n":
            res.append(line)
        else:
            pred = f"{line.strip()}{seperator}{test_prediction[cur_i]}\n"
            res.append(pred)
            cur_i += 1

    with open(output_name, mode='w') as f:
        f.writelines(res)


def split_by_mlen(batch, mlen):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(batch), mlen):
        yield batch[i:i + mlen]


def get_unpadded_pred(predicted, data_lens):
    res = []
    max_len = max(data_lens)
    predicted = predicted.cpu().detach().tolist()
    prediction_bulk = split_by_mlen(predicted, max_len) #[predicted[i::max_len] for i in range(max_len)]
    for i, bulk in enumerate(prediction_bulk):
        cur_sent = bulk[:data_lens[i]]
        res+=cur_sent

    return res




def main(mission, test_f_name, model_path, task, train_file_name, hidden_dim, lstm_hidden_dim, embedding_dim=100):
    dropout = 0.2
    sent_len = 120 if task == "ner" else 150

    chars_vocab = None
    sub_words = None

    vocab = TokenVocab(train_file_name, task)
    if mission == 'a':
        model = BiLSTMVanila(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab=vocab, dropout=dropout,
                             sent_len=sent_len)

    elif mission == 'b':
        chars_vocab = CharsVocab(train_file_name, task)
        model = BiLSTMChar(embedding_dim=embedding_dim, hidden_dim=hidden_dim,  lstm_hidden_dim=lstm_hidden_dim, vocab=vocab, chars_vocab=chars_vocab,
                           dropout=dropout, sent_len=sent_len)
    elif mission == 'c':
        sub_words = SubWords(train_file_name, task)
        model = BiLSTMSubWords(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab=vocab, sub_words=sub_words,
                               dropout=dropout, sent_len=sent_len)
    elif mission == 'd':
        chars_vocab = CharsVocab(train_file_name, task)
        model = BiLSTMConcat(embedding_dim=embedding_dim, hidden_dim=hidden_dim, lstm_hidden_dim=lstm_hidden_dim,
                             vocab=vocab, chars_vocab=chars_vocab,
                             dropout=dropout, sent_len=sent_len)
    else:
        raise ValueError(f"Not supporting repr: {mission} see help for details.")

    model.load_model(model_path)
    test_df = TokenDataFile(task=task, data_fname=test_f_name, mission=mission, vocab=vocab, sub_words=sub_words,
                            chars_vocab=chars_vocab)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test = DataLoader(test_df, batch_size=1028, collate_fn=pad_collate)
    model.eval()
    prediction = []
    for step, (data, _, data_lens, _) in tqdm(enumerate(test), total=len(test), desc=f"test data"):
        data = data.to(device)
        output = model(data, data_lens)
        _, predicted = torch.max(output, 1)
        prediction += get_unpadded_pred(predicted, data_lens)
    prediction_str = [vocab.i2label[i] for i in prediction]
    dump_test_file(prediction_str, test_f_name, vocab.separator, f"{mission}_{task}_hd{hidden_dim}_lhd{lstm_hidden_dim}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aligner model')
    parser.add_argument('repr', help="repr need to be {a,b,c,d} when\n"
                                     "a: an embedding vector.\n"
                                     "b: a character-level LSTM.\n"
                                     "c: the embeddings+subword representation used in assignment 2.\n"
                                     "d: a concatenation of (a) and (b) followed by a linear layer.")
    parser.add_argument('modelFile', help="the model file.")
    parser.add_argument('inputFile', help="The input file to test on.")
    parser.add_argument('trainFile', help="the original train file.")
    parser.add_argument('task', help="{pos, ner}")
    parser.add_argument('-lhd', '--lstm_hidden_dim', help='lstm hidden dimension value', default=200, type=int)
    parser.add_argument('-hd', '--hidden_dim', help='main hidden dimension value', default=100, type=int)

    args = parser.parse_args()
    main(mission=args.repr,
         test_f_name=args.inputFile,
         model_path=args.modelFile,
         train_file_name=args.trainFile,
         task=args.task,
         lstm_hidden_dim=args.lstm_hidden_dim,
         hidden_dim=args.hidden_dim)