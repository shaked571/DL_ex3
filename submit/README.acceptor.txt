Refael Shaked Greenfeld 305030868
Danit Yshaayahu 312434269

1. environment - activate conda environment:
    1.1 conda create -n "myenv" python=3.8
    1.2 pip install -r requirements.txt

2. files hierarchy:
    2.1. data files unzipped under data/pos and data/ner
    2.2 experiment.py under the program base dir
    2.3 vocab.txt and wordVectors.txt under data/

3. parameters values:
	train_file: path for train file
	dev_file: path for dev file
	test_file: path for test file
    optimizer: {SGD, Adam, AdamW}
    batch_size: int
    learning_rate: float
    hidden_dim: int
	embedding_dim: int
	epochs: int

4. The usage is as follow using standard argparse:
    experiment.py train_file dev_file test_file [-o --optimizer OPTIMIZER] [-b --batch_size BATCH_SIZE] 
									[-lr --l_r L_R] [-hd --hidden_dim HIDDEN_DIM] [-ed --embedding_dim EMBEDDING_DIM]
									[-e --epochs EPOCHS]
5. generate examples:
	gen_exmaples.py -p NUM_POSITIVE -m NUM_NEGATIVE -f OUTPUT_FILE_NAME [-t ADD_TAG]