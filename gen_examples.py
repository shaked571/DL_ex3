import random
import argparse
MAX_SEQ_DIGITS = 200
MAX_SEQ_LETTERS = 200
positive_letters = ["a", "b", "c", "d"]
negative_letters = ["a", "c", "b", "d"]
POSITIVE_TAG = "1"
NEGATIVE_TAG = "0"


def get_examples(positive_num, negative_num, add_tag=True):
    examples = []
    for i in range(positive_num):
        if add_tag:
            examples.append((get_example(positive_letters), POSITIVE_TAG))
        else:
            examples.append(get_example(positive_letters))

    for j in range(negative_num):
        if add_tag:
            examples.append((get_example(negative_letters), NEGATIVE_TAG))
        else:
            examples.append(get_example(negative_letters))

    random.shuffle(examples)
    return examples


def get_example(separate_letters):
    example = ""
    for i in range(len(separate_letters)):
        num_digits = random.randrange(1, MAX_SEQ_DIGITS + 1)
        for _ in range(num_digits):
            seq_num = str(random.randrange(1, 10))
            example += seq_num
        num_letters = random.randrange(1, MAX_SEQ_LETTERS)
        example += separate_letters[i]*num_letters

    num_digits = random.randrange(1, MAX_SEQ_DIGITS + 1)
    for _ in range(num_digits):
        seq_num = str(random.randrange(1, 10))
        example += seq_num
    return example


def write_examples(file_name, examples, add_tag):
    with open(file_name, mode="w") as f:
        for example in examples:
            if add_tag:
                example_str = f"{example[0]}\t{example[1]}\n"
            else:
                example_str = f"{example}\n"

            f.write(example_str)


def main(pos_num, neg_num, file_name, add_tag):
    examples = get_examples(pos_num, neg_num, add_tag)
    write_examples(file_name, examples, add_tag)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-p', '--pos', help="Number of positive examples", type=int, required=True)
    parser.add_argument('-n', '--neg', help="Number of negative examples", type=int, required=True)
    parser.add_argument('-f', '--file_name', help="Output file name", type=str, required=False)
    parser.add_argument('-t', '--add_tag', help="Add tags for the output file", action='store_true')

    args = parser.parse_args()

    main(args.pos, args.neg, args.file_name, args.add_tag)
