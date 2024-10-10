import argparse
import os
from tqdm import tqdm


def main(args):
    total_file = open(os.path.join(args.total_file, 'politeness.tsv'), 'r', encoding='utf-8')
    # split output file
    train_polite_file = open(os.path.join(args.out_path, 'train.polite.txt'), 'w', encoding='utf-8')
    train_impolite_file = open(os.path.join(args.out_path, 'train.impolite.txt'), 'w', encoding='utf-8')
    test_polite_file = open(os.path.join(args.out_path, 'test.polite.txt'), 'w', encoding='utf-8')
    test_impolite_file = open(os.path.join(args.out_path, 'test.impolite.txt'), 'w', encoding='utf-8')
    for line in tqdm(total_file):
        line_list = line.strip().split('\t')
        if line_list[2] == 'train':
            if line_list[1] == 'P_9':
                train_polite_file.write(line_list[0].strip() + '\n')
            else:
                train_impolite_file.write(line_list[0].strip() + '\n')
        else:
            if line_list[1] == 'P_9':
                test_polite_file.write(line_list[0].strip() + '\n')
            else:
                test_impolite_file.write(line_list[0].strip() + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_file", type=str)
    parser.add_argument("--out_path", type=str)
    args = parser.parse_args()
    main(args)