from transformers import RobertaTokenizer, RobertaForSequenceClassification
import tqdm, argparse, os, json
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
    tokenizer = RobertaTokenizer.from_pretrained(args.model)
    model = RobertaForSequenceClassification.from_pretrained(args.model)
    model = model.to(device)
    
    if args.base == "zero_shot":
        f_json_toxic = open(os.path.join(args.data_path, 'ParaDetox-from-toxic.jsonl'), 'r', encoding='utf-8')
        f_json_neutral = open(os.path.join(args.data_path, 'ParaDetox-from-neutral.jsonl'), 'r', encoding='utf-8')
    elif args.base == "LAPE":
        f_json_toxic = open(os.path.join(args.data_path, 'ParaDetox.toxic.perturb-neutral.jsonl'), 'r', encoding='utf-8')
        f_json_neutral = open(os.path.join(args.data_path, 'ParaDetox.neutral.perturb-toxic.jsonl'), 'r', encoding='utf-8')
    elif args.base == "LAVE":
        f_json_toxic = open(os.path.join(args.data_path, 'ParaDetox.toxic.perturb-neutral.jsonl'), 'r', encoding='utf-8')
        f_json_neutral = open(os.path.join(args.data_path, 'ParaDetox.neutral.perturb-toxic.jsonl'), 'r', encoding='utf-8')
    elif args.base == "APDN":
        pass
    elif args.base == "our":
        f_json_toxic = open(os.path.join(args.data_path, 'ParaDetox.toxic.perturb-neutral.jsonl'), 'r', encoding='utf-8')
        f_json_neutral = open(os.path.join(args.data_path, 'ParaDetox.neutral.perturb-toxic.jsonl'), 'r', encoding='utf-8')
    
    
    ### for toxic to neutral
    predict_neutral_list = []
    neural_count = 0
    
    # with open(os.path.join(args.data_path, 'ParaDetox-from-toxic.jsonl'), 'r', encoding='utf-8') as f_json:
    for json_line in f_json_toxic:
        json_dict = json.loads(json_line.strip())
        inputs = tokenizer.encode(clean_text(json_dict["output"], 'neutral'), truncation=True, max_length=512, return_tensors="pt").to(device)
        logits = model(inputs).logits
        predicted_class_id = logits.argmax().item()
        predict_label = model.config.id2label[predicted_class_id]
        predict_neutral_list.append(predict_label)
        if predict_label == "neutral":
            neural_count += 1
    # write to the results file
    with open(os.path.join(args.out_path, f'predict-neutral.txt'), 'w', encoding='utf-8') as nf:
        for pnl in predict_neutral_list:
            nf.write(pnl + '\n')
        nf.write(f'Text Style Transfer (from toxic to neutral) Accuracy : {str(neural_count/len(predict_neutral_list))}')
        
    
    ### for neutral to toxic
    
    predict_toxic_list = []
    toxic_count = 0
    # with open(os.path.join(args.data_path, 'ParaDetox-from-neutral.jsonl'), 'r', encoding='utf-8') as f_json:
    for json_line in f_json_neutral:
        json_dict = json.loads(json_line.strip())
        inputs = tokenizer.encode(clean_text(json_dict["output"], 'toxic'), truncation=True, max_length=512, return_tensors="pt").to(device)
        logits = model(inputs).logits
        predicted_class_id = logits.argmax().item()
        predict_label = model.config.id2label[predicted_class_id]
        predict_toxic_list.append(predict_label)
        if predict_label == "toxic":
            toxic_count += 1
    # write to the results file
    with open(os.path.join(args.out_path, f'predict-toxic.txt'), 'w', encoding='utf-8') as tf:
        for ptl in predict_toxic_list:
            tf.write(ptl + '\n')
        tf.write(f'Text Style Transfer (from neutral to toxic) Accuracy: {str(toxic_count/len(predict_toxic_list))}')
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="SkolkovoInstitute/roberta_toxicity_classifier")
    parser.add_argument("-b", "--base", type=str, default="baseline")
    parser.add_argument("-d", "--data_path", type=str, default="")
    parser.add_argument("-o", "--out_path", type=str, default="")
    args = parser.parse_args()
    main(args)