import argparse
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import LlamaTokenizer

"""
python preprocess_data.py \
-m meta-llama/Llama-2-7b-hf \
-d /dss/lavine/public_data/GYAFC/formal.txt \
-s formal \
-o /dss/lavine/lavine_code/TST/data
"""

def main(args):
    tokenizer = LlamaTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("text", data_files=args.data_path)

    ids=[]
    for i in tqdm(range(len(dataset['train']))):
        tokens_prompt = tokenizer(dataset['train'][i]['text'], padding="max_length", truncation=True, max_length=1024).input_ids
        ids.append(torch.tensor(tokens_prompt))

    ids=torch.stack(ids)
    final_ids = torch.reshape(ids, (-1, 1024))
    torch.save(final_ids, f'{args.output_path}/id.{args.style}.train.llama')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("-d", "--data_path", type=str, default="")
    parser.add_argument("-s", "--style", type=str, default="formal")
    parser.add_argument("-o", "--output_path", type=str, default="formal")
    args = parser.parse_args()
    main(args)
