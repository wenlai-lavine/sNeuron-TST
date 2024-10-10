import argparse
import os
from tqdm import tqdm
from datasets import load_dataset


def main(args):
    # split output file
    train_shakespeare_file = open(os.path.join(args.out_path, 'train.shakespeare.txt'), 'w', encoding='utf-8')
    train_modern_file = open(os.path.join(args.out_path, 'train.modern.txt'), 'w', encoding='utf-8')
    test_shakespeare_file = open(os.path.join(args.out_path, 'test.shakespeare.txt'), 'w', encoding='utf-8')
    test_modern_file = open(os.path.join(args.out_path, 'test.modern.txt'), 'w', encoding='utf-8')

    ## input file
    shakespeare_file = open(os.path.join(args.data_path, 'train_plays1and2_clean.original'), 'r', encoding='utf-8')
    modern_file = open(os.path.join(args.data_path, 'train_plays1and2_clean.modern'), 'r', encoding='utf-8')

    for num, (shakespeare_line, modern_line) in enumerate(zip(shakespeare_file, modern_file)):
        if num < 500:
            test_shakespeare_file.write(shakespeare_line.strip() + '\n')
            test_modern_file.write(modern_line.strip() + '\n')
        else:
            train_shakespeare_file.write(shakespeare_line.strip() + '\n')
            train_modern_file.write(modern_line.strip() + '\n')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--out_path", type=str)
    args = parser.parse_args()
    main(args)