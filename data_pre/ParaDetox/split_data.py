import argparse
import os
from tqdm import tqdm
from datasets import load_dataset


def main(args):
    # split output file
    train_toxic_file = open(os.path.join(args.out_path, 'train.toxic.txt'), 'w', encoding='utf-8')
    train_neutral_file = open(os.path.join(args.out_path, 'train.neutral.txt'), 'w', encoding='utf-8')
    test_toxic_file = open(os.path.join(args.out_path, 'test.toxic.txt'), 'w', encoding='utf-8')
    test_neutral_file = open(os.path.join(args.out_path, 'test.neutral.txt'), 'w', encoding='utf-8')

    total_dataset = load_dataset(args.data_hf)
    split_dataset = total_dataset['train'].train_test_split(test_size=0.05)
    train_dataset = split_dataset['train']
    test_dataset = split_dataset['test']
    # process train_dataset
    for data in train_dataset:
        train_toxic_file.write(data['en_toxic_comment'].strip() + '\n')
        train_neutral_file.write(data['en_neutral_comment'].strip() + '\n')
    
    # process test_dataset
    for data in test_dataset:
        test_toxic_file.write(data['en_toxic_comment'].strip() + '\n')
        test_neutral_file.write(data['en_neutral_comment'].strip() + '\n')
    

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_hf", type=str)
    parser.add_argument("--out_path", type=str)
    args = parser.parse_args()
    main(args)