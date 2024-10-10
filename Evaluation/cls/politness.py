from transformers import AutoTokenizer, AutoModelForSequenceClassification

import argparse, os, json
import torch


""" 
python code/Evaluation/cls/politness.py \
-m Genius1237/xlm-roberta-large-tydip \
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
        f_json_polite = open(os.path.join(args.data_path, 'Politness-from-polite.jsonl'), 'r', encoding='utf-8')
        f_json_impolite = open(os.path.join(args.data_path, 'Politness-from-impolite.jsonl'), 'r', encoding='utf-8')
    elif args.base == "LAPE":
        f_json_polite = open(os.path.join(args.data_path, 'Politness.polite.perturb-impolite.jsonl'), 'r', encoding='utf-8')
        f_json_impolite = open(os.path.join(args.data_path, 'Politness.impolite.perturb-polite.jsonl'), 'r', encoding='utf-8')
    elif args.base == "LAVE":
        f_json_polite = open(os.path.join(args.data_path, 'Politness.polite.perturb-impolite.jsonl'), 'r', encoding='utf-8')
        f_json_impolite = open(os.path.join(args.data_path, 'Politness.impolite.perturb-polite.jsonl'), 'r', encoding='utf-8')
    elif args.base == "APDN":
        pass
    elif args.base == "our":
        f_json_polite = open(os.path.join(args.data_path, 'Politness.polite.perturb-impolite.jsonl'), 'r', encoding='utf-8')
        f_json_impolite = open(os.path.join(args.data_path, 'Politness.impolite.perturb-polite.jsonl'), 'r', encoding='utf-8')
    
    ### for polite to impolite
    predict_impolite_list = []
    impolite_count = 0
    
    # with open(os.path.join(args.data_path, 'Politness-from-polite.jsonl'), 'r', encoding='utf-8') as f_json:
    for json_line in f_json_polite:
        json_dict = json.loads(json_line.strip())
        inputs = tokenizer(clean_text(json_dict["output"], 'impolite'), truncation=True, max_length=512, return_tensors="pt").to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()
        predict_label = model.config.id2label[predicted_class_id]
        predict_impolite_list.append(predict_label)
        if predict_label == "impolite":
            impolite_count += 1
    # write to the results file
    with open(os.path.join(args.out_path, f'predict-impolite.txt'), 'w', encoding='utf-8') as impolite_f:
        for pil in predict_impolite_list:
            impolite_f.write(pil + '\n')
        impolite_f.write(f'Text Style Transfer (from polite to impolite) Accuracy : {str(impolite_count/len(predict_impolite_list))}')
        
    ### for impolite to polite
    predict_polite_list = []
    polite_count = 0
    # with open(os.path.join(args.data_path, 'Politness-from-impolite.jsonl'), 'r', encoding='utf-8') as f_json:
    for json_line in f_json_impolite:
        json_dict = json.loads(json_line.strip())
        inputs = tokenizer(clean_text(json_dict["output"], 'polite'), truncation=True, max_length=512, return_tensors="pt").to(device)
        
        outputs = model(**inputs)
        logits = outputs.logits
        
        predicted_class_id = logits.argmax().item()
        predict_label = model.config.id2label[predicted_class_id]
        predict_polite_list.append(predict_label)
        if predict_label == "polite":
            polite_count += 1
    # write to the results file
    with open(os.path.join(args.out_path, f'predict-polite.txt'), 'w', encoding='utf-8') as ff:
        for pfl in predict_polite_list:
            ff.write(pfl + '\n')
        ff.write(f'Text Style Transfer (from impolite to polite) Accuracy: {str(polite_count/len(predict_polite_list))}')
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="SkolkovoInstitute/roberta_toxicity_classifier")
    parser.add_argument("-b", "--base", type=str, default="baseline")
    parser.add_argument("-d", "--data_path", type=str, default="")
    parser.add_argument("-o", "--out_path", type=str, default="")
    args = parser.parse_args()
    main(args)