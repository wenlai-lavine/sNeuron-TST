from transformers import AutoTokenizer, AutoModelForSequenceClassification

import argparse, os, json
import torch


""" 
python code/Evaluation/cls/politics.py \
-m m-newhauser/distilbert-political-tweets \
-b LAVE \
-d /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Baseline/LAVE/output/generate_res \
-o /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Evaluation/output_cls/LAVE

python code/Evaluation/cls/politics.py \
-m m-newhauser/distilbert-political-tweets \
-b our \
-d /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Our/output/gen_res_40000 \
-o /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Evaluation/output_cls/our_40000


"""

# import ptvsd 
# ptvsd.enable_attach(address =('0.0.0.0',5678))
# ptvsd.wait_for_attach()


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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ## load tokenizer and models
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model = model.to(device)
    
    if args.base == "zero_shot":
        f_json_democratic = open(os.path.join(args.data_path, 'Politics-from-democratic.jsonl'), 'r', encoding='utf-8')
        f_json_republican = open(os.path.join(args.data_path, 'Politics-from-republican.jsonl'), 'r', encoding='utf-8')
    elif args.base == "LAPE":
        f_json_democratic = open(os.path.join(args.data_path, 'Politics.democratic.perturb-republican.jsonl'), 'r', encoding='utf-8')
        f_json_republican = open(os.path.join(args.data_path, 'Politics.republican.perturb-democratic.jsonl'), 'r', encoding='utf-8')
    elif args.base == "LAVE":
        f_json_democratic = open(os.path.join(args.data_path, 'Politics.democratic.perturb-republican.jsonl'), 'r', encoding='utf-8')
        f_json_republican = open(os.path.join(args.data_path, 'Politics.republican.perturb-democratic.jsonl'), 'r', encoding='utf-8')
    elif args.base == "APDN":
        pass
    elif args.base == "our":
        f_json_democratic = open(os.path.join(args.data_path, 'Politics.democratic.perturb-republican.jsonl'), 'r', encoding='utf-8')
        f_json_republican = open(os.path.join(args.data_path, 'Politics.republican.perturb-democratic.jsonl'), 'r', encoding='utf-8')
    
    
    ### for democratic to republican
    predict_republican_list = []
    republican_count = 0
    
    # with open(os.path.join(args.data_path, 'Politics-from-democratic.jsonl'), 'r', encoding='utf-8') as f_json:
    for json_line in f_json_democratic:
        json_dict = json.loads(json_line.strip())
        inputs = tokenizer(clean_text(json_dict["output"], 'republican'), truncation=True, max_length=512, return_tensors="pt").to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()
        predict_label = model.config.id2label[predicted_class_id]
        predict_republican_list.append(predict_label)
        if predict_label == "Republican":
            republican_count += 1
    # write to the results file
    with open(os.path.join(args.out_path, f'predict-republican.txt'), 'w', encoding='utf-8') as republican_f:
        for pil in predict_republican_list:
            republican_f.write(pil + '\n')
        republican_f.write(f'Text Style Transfer (from democratic to republican) Accuracy : {str(republican_count/len(predict_republican_list))}')
        
    ### for republican to democratic
    predict_democratic_list = []
    democratic_count = 0
    # with open(os.path.join(args.data_path, 'Politics-from-republican.jsonl'), 'r', encoding='utf-8') as f_json:
    for json_line in f_json_republican:
        json_dict = json.loads(json_line.strip())
        inputs = tokenizer(clean_text(json_dict["output"], 'democratic'), truncation=True, max_length=512, return_tensors="pt").to(device)
        
        outputs = model(**inputs)
        logits = outputs.logits
        
        predicted_class_id = logits.argmax().item()
        predict_label = model.config.id2label[predicted_class_id]
        predict_democratic_list.append(predict_label)
        if predict_label == "Democrat":
            democratic_count += 1
    # write to the results file
    with open(os.path.join(args.out_path, f'predict-democratic.txt'), 'w', encoding='utf-8') as ff:
        for pfl in predict_democratic_list:
            ff.write(pfl + '\n')
        ff.write(f'Text Style Transfer (from republican to democratic) Accuracy: {str(democratic_count/len(predict_democratic_list))}')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="SkolkovoInstitute/roberta_toxicity_classifier")
    parser.add_argument("-b", "--base", type=str, default="baseline")
    parser.add_argument("-d", "--data_path", type=str, default="")
    parser.add_argument("-o", "--out_path", type=str, default="")
    args = parser.parse_args()
    main(args)