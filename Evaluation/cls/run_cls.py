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
    cls_dict = {
        'GYAFC': "SkolkovoInstitute/xlmr_formality_classifier",
        'ParaDetox': "SkolkovoInstitute/roberta_toxicity_classifier",
        'Politics': "m-newhauser/distilbert-political-tweets",
        'Politness': "Genius1237/xlm-roberta-large-tydip",
        'Shakespeare': "notaphoenix/shakespeare_classifier_model",
        'Yelp': "distilbert-base-uncased-finetuned-sst-2-english"
    }
    python_dict = {
        'GYAFC': "formality.py",
        'ParaDetox': "toxicity.py",
        'Politics': "politics.py",
        'Politness': "politness.py",
        'Shakespeare': "authorship.py",
        'Yelp': "sentiment.py"
    }

    for style in style_list:
        for style_name in style_dict[style]:
            for mask_style_name in style_dict[style]:
                if style_name == mask_style_name:
                    continue
                print(f"process {style} in {style_name} when mask {mask_style_name}")
                
                run_cmd = 'python ' + os.path.join(args.python_path, python_dict[style]) \
                    + ' -m ' + cls_dict[style] \
                    + ' -b ' + args.base \
                    + ' -d ' + args.data_path \
                    + ' -o ' + args.out_path
                
                print('Running Command: ' + run_cmd)
                subprocess.call(run_cmd, shell=True)
                time.sleep(5)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--python_path", type=str, default="")
    parser.add_argument("-b", "--base", type=str, default="")
    parser.add_argument("-d", "--data_path", type=str, default="")
    parser.add_argument("-o", "--out_path", type=str, default="")
    args = parser.parse_args()
    main(args)