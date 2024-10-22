import argparse
import subprocess
import os
import time


def main(args):
    style_list = ['GYAFC', 'ParaDetox', 'Politics', 'Politness', 'Shakespeare', 'Yelp']
    style_dict = {
        'GYAFC': ['formal', 'informal'],
        'ParaDetox': ['toxic', 'neutral'],
        'Politics': ['democratic', 'republican'],
        'Politness': ['polite', 'impolite'],
        'Shakespeare': ['shakespeare', 'modern'],
        'Yelp': ['positive', 'negative']
    }

    for style in style_list:
        for style_name in style_dict[style]:
            for mask_style_name in style_dict[style]:
                if style_name == mask_style_name:
                    continue
                print(f"process {style} in {style_name} when mask {mask_style_name}")
                
                run_cmd = 'python ' + args.python_file \
                    + ' -m ' + args.model \
                    + ' -a ' + args.activation_path \
                    + ' -d ' + args.data_path \
                    + ' -o ' + args.out_path \
                    + ' --style ' + style \
                    + ' --style_name ' + style_name \
                    + ' --mask_style_name ' + mask_style_name \
                    + ' --threshold ' + args.threshold
                
                print('Running Command: ' + run_cmd)
                subprocess.call(run_cmd, shell=True)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--python_file", type=str, default="")
    parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("-a", "--activation_path", type=str, default="")
    parser.add_argument("-d", "--data_path", type=str, default="")
    parser.add_argument("-o", "--out_path", type=str, default="")
    parser.add_argument("-t", "--threshold", type=str, default="5")
    args = parser.parse_args()
    main(args)