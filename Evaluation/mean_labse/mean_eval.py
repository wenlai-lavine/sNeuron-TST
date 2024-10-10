import argparse, os, json

from sentence_transformers import SentenceTransformer, util


""" 
parser.add_argument("-d", "--data_path", type=str, default="")
parser.add_argument("-o", "--out_path", type=str, default="")
parser.add_argument("-b", "--batch_size", type=str, default="")
parser.add_argument("-mp", "--model_path", type=str, default="")
parser.add_argument("-tp", "--tokenizer_path", type=str, default="")

python code/Evaluation/sim/sim_eval.py \
-d /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Baseline/zero-shot/output/llama-7b \
-o /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Evaluation/output_sim/zero-shot \
-b 8 \
-mp /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Evaluation/sim/model/sim.pt \
-tp /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Evaluation/sim/model/sim.sp.30k.model


python code/Evaluation/sim/sim_eval.py \
-d /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/gen_res/zero_shot/llama-3-8b \
-o /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/evaluation/sim/llama_8b/zero_shot \
-b zero_shot \
-bs 8 \
-mp /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Evaluation/sim/model/sim.pt \
-tp /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Evaluation/sim/model/sim.sp.30k.model

python code/Evaluation/sim/sim_eval.py \
-d /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/gen_res/LAPE/llama-3-8b \
-o /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/evaluation/sim/llama_8b/LAPE \
-b LAPE \
-bs 8 \
-mp /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Evaluation/sim/model/sim.pt \
-tp /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Evaluation/sim/model/sim.sp.30k.model

python code/Evaluation/sim/sim_eval.py \
-d /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/gen_res/LAVE/llama-3-8b \
-o /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/evaluation/sim/llama_8b/LAVE \
-b LAVE \
-bs 8 \
-mp /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Evaluation/sim/model/sim.pt \
-tp /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Evaluation/sim/model/sim.sp.30k.model

python code/Evaluation/sim/sim_eval.py \
-d /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/gen_res/gen_deact/llama-3-8b/50000 \
-o /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/evaluation/sim/llama_8b/our_deact/50000 \
-b our \
-bs 8 \
-mp /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Evaluation/sim/model/sim.pt \
-tp /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Evaluation/sim/model/sim.sp.30k.model

python code/Evaluation/sim/sim_eval.py \
-d /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/gen_res/our_dola/thres_5 \
-o /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/evaluation/sim/llama_8b/our_dola/thres_5 \
-b our \
-bs 8 \
-mp /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Evaluation/sim/model/sim.pt \
-tp /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Evaluation/sim/model/sim.sp.30k.model

python code/Evaluation/mean_labse/mean_eval.py \
-d /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/gen_res/our_dola/thres_5 \
-o /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/llama-3/evaluation/mean/llama_8b/our_dola \
-b our



"""

# import ptvsd 
# ptvsd.enable_attach(address =('0.0.0.0',5678))
# ptvsd.wait_for_attach()


def compute_cosine(pred, gold, model):
    sentences = [pred, gold]
    embeddings = model.encode(sentences, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings[0], embeddings[1])
    return cosine_scores.item()

def clean_text(txt, style):
    patten_list = ['should be rephrased as', 'as follows:', f'{style} style sentence']
    txt = txt.split('\n\n')[0]
    txt = txt.split('### Explanation')[0]
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
    
    model = SentenceTransformer('sentence-transformers/LaBSE')
    
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
            
            input_list = []
            pred_list = []
            
            with open(predict_path, 'r', encoding='utf-8') as f_json:
                for json_line in f_json:
                    json_dict = json.loads(json_line.strip())
                    input_list.append(json_dict['input'])
                    pred_list.append(clean_text(json_dict['output'], exclude_style_name))
            
            score_list = []
            # calculate the similarity
            for pre, gold in zip(pred_list, input_list):
                score = compute_cosine(pre, gold, model)
                score_list.append(score)
            
            mean_score = sum(score_list) / len(score_list)
            print(mean_score)
            
            with open(os.path.join(args.out_path, f"predict-from-{style_name}.txt"), 'w', encoding='utf-8') as wf:
                for score in score_list:
                    wf.write(str(score) + '\n')
                wf.write(f'average similarity (transfer from {style_name}) score: ' + str(mean_score))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", type=str, default="")
    parser.add_argument("-o", "--out_path", type=str, default="")
    parser.add_argument("-b", "--base", type=str, default="zero-shot")
    parser.add_argument("-bs", "--batch_size", type=int, default=8)
    parser.add_argument("-mp", "--model_path", type=str, default="")
    parser.add_argument("-tp", "--tokenizer_path", type=str, default="")
    args = parser.parse_args()
    main(args)