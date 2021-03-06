import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import BiLSTMVanila, BiLSTMChar, BiLSTMSubWords, BiLSTMConcat
from trainer import pad_collate
from vocab import TokenVocab, CharsVocab, SubWords
from DataFiles import TokenDataFile



def dump_test_file(test_prediction, test_file_path, seperator, output_name ):
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
    pred_path = f"{output_name}.tsv"
    with open(pred_path, mode='w') as f:
        f.writelines(res)



def main(mission, test_f_name, model_path, task, train_file
        ):
    embedding_dim=20
    hidden_dim=200
    dropout=0.2
    sent_len=128
    lstm_hidden_dim=50
    chars_vocab = None
    sub_words = None

    vocab = TokenVocab(train_file, task)
    if mission == 'a':
        model = BiLSTMVanila(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab=vocab, dropout=dropout,
                             sent_len=sent_len)

    elif mission == 'b':
        chars_vocab = CharsVocab(train_file, task)
        model = BiLSTMChar(embedding_dim=embedding_dim, hidden_dim=hidden_dim,  lstm_hidden_dim=lstm_hidden_dim, vocab=vocab, chars_vocab=chars_vocab,
                           dropout=dropout, sent_len=sent_len)
    elif mission == 'c':
        sub_words = SubWords(train_file, task)
        model = BiLSTMSubWords(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab=vocab, sub_words=sub_words,
                               dropout=dropout, sent_len=sent_len)
    elif mission == 'd':
        chars_vocab = CharsVocab(train_file, task)
        model = BiLSTMConcat(embedding_dim=embedding_dim, hidden_dim=hidden_dim, lstm_hidden_dim=lstm_hidden_dim,
                             vocab=vocab, chars_vocab=chars_vocab,
                             dropout=dropout, sent_len=sent_len)
    else:
        raise ValueError(f"Not supporting repr: {mission} see help for details.")
    model.load_model(model_path)
    test_df = TokenDataFile(task=task, data_fname=test_f_name, mission=mission, vocab=vocab, sub_words=sub_words,
                            chars_vocab=chars_vocab)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test = DataLoader(test_df, batch_size=128, collate_fn=pad_collate)
    model.eval()
    prediction = []
    for step, (data, _, data_lens, _) in tqdm(enumerate(test), total=len(test), desc=f"test data"):
        data = data.to(device)
        output = model(data, data_lens)
        _, predicted = torch.max(output, 1)
        prediction += predicted.tolist()
    prediction = [vocab.i2label[i] for i in prediction]
    dump_test_file(prediction, test_f_name, vocab.separator, f"{mission}_{task}_hd{hidden_dim}_lhd{lstm_hidden_dim}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aligner model')
    parser.add_argument('repr', help="repr need to be {a,b,c,d} when\n"
                                     "a: an embedding vector.\n"
                                     "b: a character-level LSTM.\n"
                                     "c: the embeddings+subword representation used in assignment 2.\n"
                                     "d: a concatenation of (a) and (b) followed by a linear layer.")
    parser.add_argument('testFile', help="The input file to test on.")
    parser.add_argument('modelFile', help="the model file.")
    parser.add_argument('-tf', '--train_file', help="the original train file.")
    parser.add_argument('-t', '--task', help="{pos, ner}")
    args = parser.parse_args()
    main(mission=args.repr,
         test_f_name=args.testFile,
         model_path=args.modelFile,
         train_file=args.train_file,
         task=args.task)