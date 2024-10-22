from transformers import AutoTokenizer, AutoModelForSequenceClassification

import argparse, os, json
import torch


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
        f_json_modern = open(os.path.join(args.data_path, 'Shakespeare-from-modern.jsonl'), 'r', encoding='utf-8')
        f_json_shakespeare = open(os.path.join(args.data_path, 'Shakespeare-from-shakespeare.jsonl'), 'r', encoding='utf-8')
    elif args.base == "LAPE":
        f_json_modern = open(os.path.join(args.data_path, 'Shakespeare.modern.perturb-shakespeare.jsonl'), 'r', encoding='utf-8')
        f_json_shakespeare = open(os.path.join(args.data_path, 'Shakespeare.shakespeare.perturb-modern.jsonl'), 'r', encoding='utf-8')
    elif args.base == "LAVE":
        f_json_modern = open(os.path.join(args.data_path, 'Shakespeare.modern.perturb-shakespeare.jsonl'), 'r', encoding='utf-8')
        f_json_shakespeare = open(os.path.join(args.data_path, 'Shakespeare.shakespeare.perturb-modern.jsonl'), 'r', encoding='utf-8')
    elif args.base == "APDN":
        pass
    elif args.base == "our":
        f_json_modern = open(os.path.join(args.data_path, 'Shakespeare.modern.perturb-shakespeare.jsonl'), 'r', encoding='utf-8')
        f_json_shakespeare = open(os.path.join(args.data_path, 'Shakespeare.shakespeare.perturb-modern.jsonl'), 'r', encoding='utf-8')
    
    
    ### for modern to shakespearean
    predict_shakespearean_list = []
    shakespearean_count = 0
    
    # with open(os.path.join(args.data_path, 'Shakespeare-from-modern.jsonl'), 'r', encoding='utf-8') as f_json:
    for json_line in f_json_modern:
        json_dict = json.loads(json_line.strip())
        inputs = tokenizer(clean_text(json_dict["output"], 'shakespearean'), truncation=True, max_length=512, return_tensors="pt").to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()
        predict_label = model.config.id2label[predicted_class_id]
        predict_shakespearean_list.append(predict_label)
        if predict_label == "shakespearean":
            shakespearean_count += 1
    # write to the results file
    with open(os.path.join(args.out_path, f'predict-shakespearean.txt'), 'w', encoding='utf-8') as shakespearean_f:
        for pil in predict_shakespearean_list:
            shakespearean_f.write(pil + '\n')
        shakespearean_f.write(f'Text Style Transfer (from modern to shakespearean) Accuracy : {str(shakespearean_count/len(predict_shakespearean_list))}')
        
    ### for shakespearean to modern
    predict_modern_list = []
    modern_count = 0
    # with open(os.path.join(args.data_path, 'Shakespeare-from-shakespeare.jsonl'), 'r', encoding='utf-8') as f_json:
    for json_line in f_json_shakespeare:
        json_dict = json.loads(json_line.strip())
        inputs = tokenizer(clean_text(json_dict["output"], 'modern'), truncation=True, max_length=512, return_tensors="pt").to(device)
        
        outputs = model(**inputs)
        logits = outputs.logits
        
        predicted_class_id = logits.argmax().item()
        predict_label = model.config.id2label[predicted_class_id]
        predict_modern_list.append(predict_label)
        if predict_label == "modern":
            modern_count += 1
    # write to the results file
    with open(os.path.join(args.out_path, f'predict-modern.txt'), 'w', encoding='utf-8') as ff:
        for pfl in predict_modern_list:
            ff.write(pfl + '\n')
        ff.write(f'Text Style Transfer (from shakespearean to modern) Accuracy: {str(modern_count/len(predict_modern_list))}')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="SkolkovoInstitute/roberta_toxicity_classifier")
    parser.add_argument("-b", "--base", type=str, default="baseline")
    parser.add_argument("-d", "--data_path", type=str, default="")
    parser.add_argument("-o", "--out_path", type=str, default="")
    args = parser.parse_args()
    main(args)