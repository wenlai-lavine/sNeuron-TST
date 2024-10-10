import argparse
import os
from tqdm import tqdm
import csv
import random


def filter_txt(text):
    if text.isascii():
        if 5 < len(text.split(' ')) < 80:
            return True
        else:
            return False
    else:
        # 删掉
        return False


def main(args):
    # split output file
    train_democratic_file = open(os.path.join(args.out_path, 'train.democratic.txt'), 'w', encoding='utf-8')
    train_republican_file = open(os.path.join(args.out_path, 'train.republican.txt'), 'w', encoding='utf-8')
    test_democratic_file = open(os.path.join(args.out_path, 'test.democratic.txt'), 'w', encoding='utf-8')
    test_republican_file = open(os.path.join(args.out_path, 'test.republican.txt'), 'w', encoding='utf-8')

    total_republican_list = []
    total_democratic_list = []
    filter_count = 0

    ## input file
    with open(os.path.join(args.data_path, 'facebook_congress_responses.csv'), 'r', encoding='utf-8') as input_file:
        for row in tqdm(csv.DictReader(input_file, skipinitialspace=True)):
            tmp_txt = row['response_text']
            congress = row['op_category']
            if filter_txt(tmp_txt):
                if congress == 'Congress_Republican':
                    total_republican_list.append(tmp_txt)
                elif congress == 'Congress_Democratic':
                    total_democratic_list.append(tmp_txt)
                else:
                    continue
            else:
                filter_count += 1
                continue
    

    ## 打乱list
    random.shuffle(total_republican_list)
    random.shuffle(total_democratic_list)

    for num, rep_line in enumerate(total_republican_list):
        if num < 1000:
            test_republican_file.write(rep_line.strip() + '\n')
        else:
            train_republican_file.write(rep_line.strip() + '\n')
    
    for num, dem_line in enumerate(total_democratic_list):
        if num < 1000:
            test_democratic_file.write(dem_line.strip() + '\n')
        else:
            train_democratic_file.write(dem_line.strip() + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--out_path", type=str)
    args = parser.parse_args()
    main(args)