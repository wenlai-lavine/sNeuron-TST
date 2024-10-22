import argparse
import json
import os
from tqdm import tqdm
import heapq

import torch
from vllm import LLM, SamplingParams
import torch.nn.functional as F
from types import MethodType

from contra_neurons import Con_Neu


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "translate_instruct": "Please translate the following {source_lang} sentence to {target_lang}: {source_sentence}"
}


prompt_dict = {
        'GYAFC': {
            'formal': 'Please transfer the following formal style sentence into an informal style sentence and maintain the meaning of the sentence.\nFormal Style Setence: {text}\nPlease only return the informal style sentence.',
            'informal': 'Please transfer the following informal style sentence into a formal style sentence and maintain the meaning of the sentence.\nInformal Style Sentence: {text}\nPlease only return the formal style sentence.'
        },
        'ParaDetox': {
            'toxic': 'Please transfer the following toxic style sentence into a neutral style sentence and maintain the meaning of the sentence.\nToxic Style Setence: {text}\nPlease only return the neutral style sentence.',
            'neutral': 'Please transfer the following neutral style sentence into a toxic style sentence and maintain the meaning of the sentence.\nNeutral Style Setence: {text}\nPlease only return the toxic style sentence.'
        },
        'Politics': {
            'democratic': 'Please transfer the following democratic style sentence into a republican style sentence and maintain the meaning of the sentence.\nDemocratic Style Setence: {text}\nPlease only return the republican style sentence.',
            'republican': 'Please transfer the following republican style sentence into a democratic style sentence and maintain the meaning of the sentence.\nRepublican Style Setence: {text}\nPlease only return the democratic style sentence.'
        },
        'Politness': {
            'polite': 'Please transfer the following polite style sentence into an impolite style sentence and maintain the meaning of the sentence.\nPolite Style Setence: {text}\nPlease only return the impolite style sentence.',
            'impolite': 'Please transfer the following impolite style sentence into a polite style sentence and maintain the meaning of the sentence.\nImpolite Style Setence: {text}\nPlease only return the polite style sentence.'
        },
        'Shakespeare': {
            'shakespeare': 'Please transfer the following shakespeare style sentence into a modern style sentence and maintain the meaning of the sentence.\nShakespeare Style Setence: {text}\nPlease only return the modern style sentence.',
            'modern': 'Please transfer the following modern style sentence into a shakespeare style sentence and maintain the meaning of the sentence.\nModern Style Setence: {text}\nPlease only return the shakespeare style sentence.'
        },
        'Yelp': {
            'positive': 'Please transfer the following positive style sentence into a negative style sentence and maintain the meaning of the sentence.\nPositive Style Setence: {text}\nPlease only return the negative style sentence.',
            'negative': 'Please transfer the following negative style sentence into a positive style sentence and maintain the meaning of the sentence.\nNegative Style Setence: {text}\nPlease only return the positive style sentence.'
        }
    }


def load_dataset(data_path, style, style_name):
    texts = []
    original_texts = []
    with open(data_path, 'r') as f:
        for line in f:
            prompt_text = PROMPT_DICT['prompt_no_input'].format(instruction=prompt_dict[style][style_name].format(text=line.strip()))
            texts.append(prompt_text)
            original_texts.append(line.strip())
    return texts, original_texts


def main(args):
    llm = Con_Neu(model_name=args.model, device="cuda", num_gpus=1, max_gpu_memory=27)
    
    is_llama = bool(args.model.lower().find("llama") >= 0)
    
    ## 
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)
    
    
    def factory(mask):
        def llama_forward(self, x):
            gate = self.gate_proj(x)
            activation = F.silu(gate)
            activation.index_fill_(2, mask, 0)
            x = activation * self.up_proj(x)
            x = self.down_proj(x)
            return x

        def bloom_forward(self, x: torch.Tensor):
            x, _ = self.dense_h_to_4h(x)
            x = self.gelu_impl(x)
            x.index_fill_(2, mask, 0)
            x, _ = self.dense_4h_to_h(x)
            return x

        if is_llama:
            return llama_forward
        else:
            return bloom_forward
    
    
    style_dict = {
        'GYAFC': ['formal', 'informal'],
        'ParaDetox': ['toxic', 'neutral'],
        'Politics': ['democratic', 'republican'],
        'Politness': ['polite', 'impolite'],
        'Shakespeare': ['shakespeare', 'modern'],
        'Yelp': ['positive', 'negative']
    }
    
    activation_masks = torch.load(f"{args.activation_mask}/{args.style}.llama-7b")
    
    mask_activation = activation_masks[style_dict[args.style].index(args.mask_style_name)]

    for i, mask_act in enumerate(mask_activation):
        if is_llama:
            obj = llm.model.model.layers[i].mlp
        else:
            obj = llm.model.model.transformer.h[i].mlp
        obj.forward = MethodType(factory(mask_act.to('cuda')), obj)
    
    print(f"start to process the dataset: {args.style_name} ---mask style {args.mask_style_name} ... ...")
    print('load dataset ... ...')
    
    mode = "Con_Neu"
    if args.repetition_penalty is None:
        args.repetition_penalty = 1.2
    
    datasets, orig_datasets = load_dataset(os.path.join(args.data_path, args.style, f'test.{args.style_name}.txt'), args.style, args.style_name)
    
    outputs = []
    
    for sample in tqdm(datasets):
        generate_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=args.do_sample, top_p=args.top_p, top_k=args.top_k, temperature=args.temperature, repetition_penalty=args.repetition_penalty, mode=mode, relative_top=args.relative_top)
        model_completion, _ = llm.generate(sample, **generate_kwargs)
        
        final_output = model_completion.replace('\n', '', 5)
        outputs.append(final_output)

    with open(os.path.join(args.out_path, f"{args.style}.{args.style_name}.perturb-{args.mask_style_name}.jsonl"), 'w', encoding='utf-8') as f:
        for t, ori, o in zip(datasets, orig_datasets, outputs):
            out = {
                "source_style": args.style_name,
                "input": ori,
                "instruction": t,
                "output": o,
                
            }
            f.writelines(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()           
                
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("-a", "--activation_mask", type=str, default="")
    parser.add_argument("-d", "--data_path", type=str, default="")
    parser.add_argument("-o", "--out_path", type=str, default="")
    parser.add_argument("--style", type=str, default="")
    parser.add_argument("--style_name", type=str, default="")
    parser.add_argument("--mask_style_name", type=str, default="")
    
    ## dola setting
    parser.add_argument("--early-exit-layers", type=str, default="-1")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--total-shard", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--relative_top", type=float, default=0.1)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--do_shuffle", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retry", type=int, default=1)
    
    
    args = parser.parse_args()
    main(args)