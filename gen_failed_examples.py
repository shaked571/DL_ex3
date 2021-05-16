import random
import argparse
from gen_examples import POSITIVE_TAG, NEGATIVE_TAG, write_examples


def get_ordered_numbers_example(pos):
    ordered_numbers = "123456789"
    if pos:
        ordered_numbers = ordered_numbers[::-1]
    return ordered_numbers*10


def get_palindrome_example(pos):
    seq_len = 50
    gen_str = ""
    for _ in range(seq_len):
        curr_num = random.randrange(1, 10)
        gen_str += str(curr_num)

    if pos:
        return gen_str + gen_str[::-1]
    return gen_str + gen_str


def get_sum_example(pos):
    seq_sum = 105
    if pos:
        seq_sum = 100

    sum_str = ""
    while seq_sum > 9:
        curr_num = random.randrange(1, 10)
        sum_str += str(curr_num)
        seq_sum -= curr_num
    sum_str += str(seq_sum)
    return sum_str


def get_examples(examples_num, example_func):
    examples = []
    for i in range(int(examples_num/2)):
        examples.append((example_func(True), POSITIVE_TAG))
        examples.append((example_func(False), NEGATIVE_TAG))

    random.shuffle(examples)
    return examples


def main(examples_num, examples_type, file_name):
    if examples_type == "sum":
        example_func = get_sum_example
    elif examples_type == "palindrome":
        example_func = get_palindrome_example
    elif examples_type == "ordered":
        example_func = get_ordered_numbers_example
    else:
        print("examples type is not valid! choose: sum/palindrome/ordered")
        return
    examples = get_examples(examples_num, example_func)
    write_examples(file_name, examples, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_name', help="Output file name", type=str, required=True)
    parser.add_argument('-n', '--examples_num', help="Number of examples: half positive and half negative", type=int, required=True)
    parser.add_argument('-t', '--examples_type', help="Type of examples: sum, palindrome or ordered", type=str, required=True)

    args = parser.parse_args()

    main(examples_num=args.examples_num,
         examples_type=args.examples_type,
         file_name=args.file_name)
