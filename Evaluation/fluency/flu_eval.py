import argparse, os, json
from fluency import do_cola_eval, calc_gpt_ppl


""" 
parser.add_argument("-d", "--data_path", type=str, default="")
parser.add_argument("-o", "--out_path", type=str, default="")
parser.add_argument("-b", "--batch_size", type=int, default="")
parser.add_argument("-mp", "--model_path", type=str, default="")

python code/Evaluation/fluency/flu_eval.py \
-d /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Baseline/zero-shot/output/llama-7b \
-o /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Evaluation/output_flu/zero-shot \
-mp openai-community/gpt2-large

python code/Evaluation/fluency/flu_eval.py \
-b zero_shot \
-d /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/gen_res/zero_shot/llama-3-8b \
-o /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/evaluation/flu/llama_8b/zero_shot \
-mp openai-community/gpt2-large

python code/Evaluation/fluency/flu_eval.py \
-b LAPE \
-d /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/gen_res/LAPE/llama-3-8b \
-o /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/evaluation/flu/llama_8b/LAPE \
-mp openai-community/gpt2-large

python code/Evaluation/fluency/flu_eval.py \
-b LAVE \
-d /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/gen_res/LAVE/llama-3-8b \
-o /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/evaluation/flu/llama_8b/LAVE \
-mp openai-community/gpt2-large

python code/Evaluation/fluency/flu_eval.py \
-b our \
-d /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/gen_res/gen_deact/llama-3-8b/50000 \
-o /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/evaluation/flu/llama_8b/our_deact/50000 \
-mp openai-community/gpt2-large


python code/Evaluation/fluency/flu_eval.py \
-b our \
-d /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/gen_res/act_deact_src_tgt_analysis/gen_act_src_deact_tgt \
-o /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/evaluation/flu/llama_8b/act_deact_src_tgt_analysis/act_src_deact_tgt \
-mp openai-community/gpt2-large



"""

# import ptvsd 
# ptvsd.enable_attach(address =('0.0.0.0',5678))
# ptvsd.wait_for_attach()

def clean_text(txt):
    patten_list = ['should be rephrased as', 'as follows:']
    txt = txt.split('\n\n')[0]
    patten_str = ""
    for patten in patten_list:
        if patten in txt:
            patten_str = patten
            break
    if patten_str:
        txt = txt.split(patten_str)[1]
    ## 去掉```
    txt.replace('```', '', 5)
    txt.replace('\\', '', 5)
    
    return txt


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
            print(f'process {style} --- --- {style_name}')
            exclude_style_name = [x for x in style_dict[style] if x != style_name][0]

            if args.base == "zero_shot":
                predict_path = os.path.join(args.data_path, f"{style}-from-{style_name}.jsonl")
            elif args.base == "LAPE":
                predict_path = os.path.join(args.data_path, f"{style}.{style_name}.perturb-{exclude_style_name}.jsonl")
            elif args.base == "LAVE":
                predict_path = os.path.join(args.data_path, f"{style}.{style_name}.perturb-{exclude_style_name}.jsonl")
            elif args.base == "APDN":
                pass
            elif args.base == "our":
                predict_path = os.path.join(args.data_path, f"{style}.{style_name}.perturb-{exclude_style_name}.jsonl")
    
            pred_list = []
            
            with open(predict_path, 'r', encoding='utf-8') as f_json:
                for json_line in f_json:
                    json_dict = json.loads(json_line.strip())
                    pred_list.append(clean_text(json_dict['output']))
            
            # calculate the similarity
            # cola_stats = do_cola_eval(args, pred_list)
            # cola_acc = sum(cola_stats) / len(pred_list)
            # print(cola_acc)
            
            token_ppl, avg_ppl = calc_gpt_ppl(args, pred_list)
            # print(token_ppl)
            
            with open(os.path.join(args.out_path, f"predict-from-{style_name}.txt"), 'w', encoding='utf-8') as wf:
                for score in token_ppl:
                    wf.write(str(score) + '\n')
                wf.write(f'average ppl (transfer from {style_name}) score: ' + str(avg_ppl))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base", type=str, default="zero_shot")
    parser.add_argument("-d", "--data_path", type=str, default="")
    parser.add_argument("-o", "--out_path", type=str, default="")
    parser.add_argument("-mp", "--model_path", type=str, default="")
    args = parser.parse_args()
    main(args)