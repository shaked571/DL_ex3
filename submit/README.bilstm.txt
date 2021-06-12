Refael Shaked Greenfeld 305030868
Danit Yshaayahu 312434269

1. environment - activate conda environment (like part 1):
    1.1 conda create -n "myenv" python=3.8
    1.2 pip install -r requirements.txt

2. files hierarchy:
    2.1. data files unzipped under data/pos and data/ner
    2.2 experiment.py under the program base dir
    2.3 vocab.txt and wordVectors.txt under data/

3. additional parameters (use --help for explanations):
	3.1 training:
			devFile: path for dev file
			task: {pos, ner}
			optimizer: {SGD, Adam, AdamW}
			epochs: int
			sent_len: int (maximum sentence length)
			batch_size: int
			learning_rate: float
			drop_out: float
			hidden_dim: int
			embedding_dim: int
			lstm_hidden_dim: int (for char's LSTM in part b and d)
			
	3.2 predict:
			trainFile: path for train file
			task: {pos, ner}
			hidden_dim: int
			lstm_hidden_dim: int (for char's LSTM in part b and d)

4. The usage is as follow using standard argparse:
	4.1 train:
		bilstmTrain.py repr trainFile modelFile dev_file task [-o --optimizer OPTIMIZER] [-b --batch_size BATCH_SIZE] 
										[-lr --l_r L_R] [-hd --hidden_dim HIDDEN_DIM] [-lhd --lstm_hidden_dim HIDDEN_DIM] 
										[-ed --embedding_dim EMBEDDING_DIM] [-e --epochs EPOCHS] [-s --sent_len SENT_LEN] 
										[-l --learning_rate LEARNING_RATE] [-do --drop_out DROP_OUT]
	4.2 predict:
		bilstmPredict.py repr modelFile inputFile trainFile task [-hd --hidden_dim HIDDEN_DIM] [-lhd --lstm_hidden_dim HIDDEN_DIM]
		4.2.1 hidden_dim and lstm_hidden_dim shuld be same as in training (in default they the same)
			   