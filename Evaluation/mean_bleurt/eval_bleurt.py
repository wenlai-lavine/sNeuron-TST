import argparse, os, json

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


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
    
    device = torch.device('cuda')
    
    tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-base-128")
    model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-base-128")
    model = model.to(device)
    model.eval()

    
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
                with torch.no_grad():
                    input_ids = tokenizer(gold, pre, truncation=True, max_length=128, return_tensors='pt').to(device)
                    scores = model(**input_ids)[0].squeeze()
                    score_list.append(scores.tolist())
            
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