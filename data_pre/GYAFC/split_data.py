import argparse
import os
from tqdm import tqdm
from datasets import load_dataset


def main(args):
    # split output file
    train_formal_file = open(os.path.join(args.out_path, 'train.formal.txt'), 'w', encoding='utf-8')
    train_informal_file = open(os.path.join(args.out_path, 'train.informal.txt'), 'w', encoding='utf-8')
    test_formal_file = open(os.path.join(args.out_path, 'test.formal.txt'), 'w', encoding='utf-8')
    test_informal_file = open(os.path.join(args.out_path, 'test.informal.txt'), 'w', encoding='utf-8')

    ## input file
    formal_file = open(os.path.join(args.data_path, 'formal.txt'), 'r', encoding='utf-8')
    informal_file = open(os.path.join(args.data_path, 'informal.txt'), 'r', encoding='utf-8')

    for num, (formal_line, informal_line) in enumerate(zip(formal_file, informal_file)):
        if num < 500:
            test_formal_file.write(formal_line.strip() + '\n')
            test_informal_file.write(informal_line.strip() + '\n')
        else:
            train_formal_file.write(formal_line.strip() + '\n')
            train_informal_file.write(informal_line.strip() + '\n')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--out_path", type=str)
    args = parser.parse_args()
    main(args)