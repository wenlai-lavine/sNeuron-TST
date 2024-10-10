from transformers import XLMRobertaTokenizerFast, XLMRobertaForSequenceClassification

import tqdm, argparse, os, json
import torch


""" 
python code/Evaluation/cls/formality.py \
-m SkolkovoInstitute/xlmr_formality_classifier \
-d /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Baseline/zero-shot/output/llama-7b \
-o /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Evaluation/output/zero-shot

python code/Evaluation/cls/formality.py \
-m SkolkovoInstitute/xlmr_formality_classifier \
-b LAPE \
-d /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Baseline/LAPE/output/generate_res \
-o /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Evaluation/output_cls/LAPE

python code/Evaluation/cls/formality.py \
-m SkolkovoInstitute/xlmr_formality_classifier \
-b LAVE \
-d /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Baseline/LAVE/output/generate_res \
-o /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Evaluation/output_cls/LAVE

python code/Evaluation/cls/formality.py \
-m SkolkovoInstitute/xlmr_formality_classifier \
-b our \
-d /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Our/output/gen_res_40000 \
-o /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Evaluation/output_cls/our_40000

python code/Evaluation/cls/formality.py \
-m SkolkovoInstitute/xlmr_formality_classifier \
-b our \
-d /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Our/output/gen_dola_layer_30000 \
-o /dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/lavine/lavine_code/TST/code/Evaluation/output_cls/our_dola_layer_30000

python code/Evaluation/cls/formality.py \
-m SkolkovoInstitute/xlmr_formality_classifier \
-b our \
-d /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/our_gen_con_neu/30000 \
-o /dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/lavine/output/evaluation/cls/30000

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
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(args.model)
    model = XLMRobertaForSequenceClassification.from_pretrained(args.model)
    model = model.to(device)
    
    ## 
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)
    
    
    if args.base == "zero_shot":
        f_json_formal_informal = open(os.path.join(args.data_path, 'GYAFC-from-formal.jsonl'), 'r', encoding='utf-8')
        f_json_informal_formal = open(os.path.join(args.data_path, 'GYAFC-from-informal.jsonl'), 'r', encoding='utf-8')
    elif args.base == "LAPE":
        f_json_formal_informal = open(os.path.join(args.data_path, 'GYAFC.formal.perturb-informal.jsonl'), 'r', encoding='utf-8')
        f_json_informal_formal = open(os.path.join(args.data_path, 'GYAFC.informal.perturb-formal.jsonl'), 'r', encoding='utf-8')
    elif args.base == "LAVE":
        f_json_formal_informal = open(os.path.join(args.data_path, 'GYAFC.formal.perturb-informal.jsonl'), 'r', encoding='utf-8')
        f_json_informal_formal = open(os.path.join(args.data_path, 'GYAFC.informal.perturb-formal.jsonl'), 'r', encoding='utf-8')
    elif args.base == "APDN":
        pass
    elif args.base == "our":
        # f_json_formal_formal = open(os.path.join(args.data_path, 'GYAFC.formal.perturb-formal.jsonl'), 'r', encoding='utf-8')
        f_json_formal_informal = open(os.path.join(args.data_path, 'GYAFC.formal.perturb-informal.jsonl'), 'r', encoding='utf-8')
        # f_json_informal_informal = open(os.path.join(args.data_path, 'GYAFC.informal.perturb-informal.jsonl'), 'r', encoding='utf-8')
        f_json_informal_formal = open(os.path.join(args.data_path, 'GYAFC.informal.perturb-formal.jsonl'), 'r', encoding='utf-8')
        
    
    ### for formal to informal
    # predict_informal_list = []
    # informal_count = 0
    # # with open(os.path.join(args.data_path, 'GYAFC-from-formal.jsonl'), 'r', encoding='utf-8') as f_json:
    # for json_line in f_json_formal_formal:
    #     json_dict = json.loads(json_line.strip())
    #     inputs = tokenizer.encode(clean_text(json_dict["output"], 'informal'), truncation=True, max_length=512, return_tensors="pt")
    #     logits = model(inputs).logits
    #     predicted_class_id = logits.argmax().item()
    #     predict_label = model.config.id2label[predicted_class_id]
    #     predict_informal_list.append(predict_label)
    #     if predict_label == "informal":
    #         informal_count += 1
    # # write to the results file
    # with open(os.path.join(args.out_path, f'predict-formal-formal.txt'), 'w', encoding='utf-8') as informal_f:
    #     for pil in predict_informal_list:
    #         informal_f.write(pil + '\n')
    #     informal_f.write(f'Text Style Transfer (from formal to informal) Accuracy : {str(informal_count/len(predict_informal_list))}')
        
    
    predict_informal_list = []
    informal_count = 0
    for json_line in f_json_formal_informal:
        json_dict = json.loads(json_line.strip())
        inputs = tokenizer.encode(clean_text(json_dict["output"], 'informal'), truncation=True, max_length=512, return_tensors="pt").to(device)
        logits = model(inputs).logits
        predicted_class_id = logits.argmax().item()
        predict_label = model.config.id2label[predicted_class_id]
        predict_informal_list.append(predict_label)
        if predict_label == "informal":
            informal_count += 1
    # write to the results file
    with open(os.path.join(args.out_path, f'predict-formal-informal.txt'), 'w', encoding='utf-8') as informal_f:
        for pil in predict_informal_list:
            informal_f.write(pil + '\n')
        informal_f.write(f'Text Style Transfer (from formal to informal) Accuracy : {str(informal_count/len(predict_informal_list))}')    
    
    
        
    ### for informal to formal
    # predict_formal_list = []
    # formal_count = 0
    # # with open(os.path.join(args.data_path, 'GYAFC-from-informal.jsonl'), 'r', encoding='utf-8') as f_json:
    # for json_line in f_json_informal_informal:
    #     json_dict = json.loads(json_line.strip())
    #     inputs = tokenizer.encode(clean_text(json_dict["output"], 'formal'), truncation=True, max_length=512, return_tensors="pt")
    #     logits = model(inputs).logits
    #     predicted_class_id = logits.argmax().item()
    #     predict_label = model.config.id2label[predicted_class_id]
    #     predict_formal_list.append(predict_label)
    #     if predict_label == "formal":
    #         formal_count += 1
    # # write to the results file
    # with open(os.path.join(args.out_path, f'predict-informal-informal.txt'), 'w', encoding='utf-8') as ff:
    #     for pfl in predict_formal_list:
    #         ff.write(pfl + '\n')
    #     ff.write(f'Text Style Transfer (from informal to formal) Accuracy: {str(formal_count/len(predict_formal_list))}')
    
    
    predict_formal_list = []
    formal_count = 0
    # with open(os.path.join(args.data_path, 'GYAFC-from-informal.jsonl'), 'r', encoding='utf-8') as f_json:
    for json_line in f_json_informal_formal:
        json_dict = json.loads(json_line.strip())
        inputs = tokenizer.encode(clean_text(json_dict["output"], 'formal'), truncation=True, max_length=512, return_tensors="pt").to(device)
        logits = model(inputs).logits
        predicted_class_id = logits.argmax().item()
        predict_label = model.config.id2label[predicted_class_id]
        predict_formal_list.append(predict_label)
        if predict_label == "formal":
            formal_count += 1
    # write to the results file
    with open(os.path.join(args.out_path, f'predict-informal-formal.txt'), 'w', encoding='utf-8') as ff:
        for pfl in predict_formal_list:
            ff.write(pfl + '\n')
        ff.write(f'Text Style Transfer (from informal to formal) Accuracy: {str(formal_count/len(predict_formal_list))}')
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base", type=str, default="baseline")
    parser.add_argument("-m", "--model", type=str, default="SkolkovoInstitute/roberta_toxicity_classifier")
    parser.add_argument("-d", "--data_path", type=str, default="")
    parser.add_argument("-o", "--out_path", type=str, default="")
    args = parser.parse_args()
    main(args)