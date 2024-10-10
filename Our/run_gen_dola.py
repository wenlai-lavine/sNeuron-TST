import argparse
import subprocess
import os
import time

"""
nohup python code/Our/run_gen_dola.py \
-p /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Our/generate_dola.py \
-m meta-llama/Llama-2-7b-hf \
-a /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/our_nerons/final_neurons \
-d /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/data \
-o /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/our_final_gen \
> log.txt &

nohup python code/Our/run_gen_dola.py \
-t 9 \
-p /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Our/generate_dola.py \
-m meta-llama/Meta-Llama-3-8B \
-a /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/neurons/our/final \
-d /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/data \
-o /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/gen_res/our_dola/thres_9 \
> /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/gen_res/our_dola/thres_9/log.txt &


nohup python code/Our/run_gen_dola.py \
-t 5 \
-p /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Our/generate_dola.py \
-m meta-llama/Meta-Llama-3-8B \
-a /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/neurons/our/final \
-d /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/data \
-o /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/gen_res/our_dola_min/thres_5 \
> /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/gen_res/our_dola_min/thres_5/log.txt &


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

    for style in style_list:
        for style_name in style_dict[style]:
            for mask_style_name in style_dict[style]:
                if style_name == mask_style_name:
                    continue
                print(f"process {style} in {style_name} when mask {mask_style_name}")

                """ 
                nohup python code/Our/gen_contrast_neurons.py \
                -m meta-llama/Llama-2-7b-hf \
                -a /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/our_nerons/30000 \
                -d /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/data \
                -o /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/our_gen_con_neu/30000 \
                > log.txt &
                """
                
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
                time.sleep(30)
            

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