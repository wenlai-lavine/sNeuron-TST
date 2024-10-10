import argparse
import subprocess
import os
import time

"""
nohup python code/Evaluation/cls/run_cls.py \
-b zero_shot \
-p /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Evaluation/cls \
-d /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/gen_res/zero_shot/llama-3-8b \
-o /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/evaluation/cls/llama_8b/zero_shot \
> log_zero_shot_cls.txt &

nohup python code/Evaluation/cls/run_cls.py \
-b LAPE \
-p /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Evaluation/cls \
-d /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/gen_res/LAPE/llama-3-8b \
-o /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/evaluation/cls/llama_8b/LAPE \
> log_LAPE_cls.txt &

nohup python code/Evaluation/cls/run_cls.py \
-b LAVE \
-p /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Evaluation/cls \
-d /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/gen_res/LAVE/llama-3-8b \
-o /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/evaluation/cls/llama_8b/LAVE \
> log_LAVE_cls.txt &

nohup python code/Evaluation/cls/run_cls.py \
-b our \
-p /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Evaluation/cls \
-d /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/gen_res/gen_deact/llama-3-8b/5000 \
-o /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/evaluation/cls/llama_8b/our_deact/5000 \
> log_50000.txt &

nohup python code/Evaluation/cls/run_cls.py \
-b our \
-p /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Evaluation/cls \
-d /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/gen_res/our_dola/thres_8 \
-o /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/evaluation/cls/llama_8b/our_dola/thres_8 \
> /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/evaluation/cls/llama_8b/our_dola/thres_8/log.txt &


python code/Evaluation/cls/run_cls.py \
-b our \
-p /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Evaluation/cls \
-d /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/Analysis/inter_motivation/non_inter_res \
-o /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/Analysis/inter_motivation/eval/non_inter/cls \
> /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/evaluation/cls/llama_8b/our_dola/thres_8/log.txt &

python code/Evaluation/cls/run_cls.py \
-b our \
-p /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Evaluation/cls \
-d /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/Analysis/inter_motivation/task_res \
-o /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/Analysis/inter_motivation/eval/task/cls

python code/Evaluation/cls/run_cls.py \
-b our \
-p /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Evaluation/cls \
-d /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/gen_res/our_dola_min/thres_5 \
-o /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/evaluation/cls/llama_8b/our_dola_min/thres_5


"""

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