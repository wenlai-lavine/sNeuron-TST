import argparse
from types import MethodType

import torch
from vllm import LLM, SamplingParams


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("-s", "--style", type=str, default="formal")
parser.add_argument("-d", "--data_path", type=str, default="")
args = parser.parse_args()

is_llama = bool(args.model.lower().find('llama') >= 0)
model = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True)

max_length = model.llm_engine.model_config.max_model_len
num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers
intermediate_size = model.llm_engine.model_config.hf_config.intermediate_size if is_llama else model.llm_engine.model_config.hf_config.hidden_size * 4

sum1 = torch.zeros(num_layers, intermediate_size).to('cuda')
sum2 = torch.zeros(num_layers, intermediate_size).to('cuda')
sum3 = torch.zeros(num_layers, intermediate_size).to('cuda')
sum4 = torch.zeros(num_layers, intermediate_size).to('cuda')
over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to('cuda')

def factory(idx):
    def llama_forward(self, x):
        gate_up, _ = self.gate_up_proj(x)  # b, l, 2i
        i = gate_up.size(-1)
        gate_up[:, :, : i // 2] = torch.nn.SiLU()(gate_up[:, :, : i // 2])
        activation = gate_up[:, :, : i // 2].float() # b, l, i
        sum1[idx, :] += activation.sum(dim=(0,1))
        sum2[idx, :] += activation.pow(2).sum(dim=(0,1))
        sum3[idx, :] += activation.pow(3).sum(dim=(0,1))
        sum4[idx, :] += activation.pow(4).sum(dim=(0,1))
        over_zero[idx, :] += (activation > 0).sum(dim=(0,1))
        x = gate_up[:, :, : i // 2] * gate_up[:, :, i // 2 :]
        x, _ = self.down_proj(x)
        return x

    def bloom_forward(self, x: torch.Tensor):
        x, _ = self.dense_h_to_4h(x)
        x = self.gelu_impl(x)
        activation = x.float()
        sum1[idx, :] += activation.sum(dim=(0,1))
        sum2[idx, :] += activation.pow(2).sum(dim=(0,1))
        sum3[idx, :] += activation.pow(3).sum(dim=(0,1))
        sum4[idx, :] += activation.pow(4).sum(dim=(0,1))
        over_zero[idx, :] += (activation > 0).sum(dim=(0,1))
        x, _ = self.dense_4h_to_h(x)
        return x

    if is_llama:
        return llama_forward
    else:
        return bloom_forward

for i in range(num_layers):
    if is_llama:
        obj = model.llm_engine.driver_worker.model_runner.model.model.layers[i].mlp
    else:
        obj = model.llm_engine.driver_worker.model_runner.model.transformer.h[i].mlp
    obj.forward = MethodType(factory(i), obj)

style = args.style
if is_llama:
    ids = torch.load(f'{args.data_path}/id.{style}.train.llama')
else:
    ids = torch.load(f'{args.data_path}/id.{style}.train.bloom')
l = ids.size(0)
l = min(l, 99999744) // max_length * max_length
input_ids = ids[:l].reshape(-1, max_length)

output = model.generate(prompt_token_ids=input_ids.tolist(), sampling_params=SamplingParams(max_tokens=1))

output = dict(n=l, sum1=sum1.to('cpu'), sum2=sum2.to('cpu'), sum3=sum3.to('cpu'), sum4=sum4.to('cpu'), over_zero=over_zero.to('cpu'))

if is_llama:
    torch.save(output, f'{args.data_path}/activation.{style}.train.llama-7b')
else:
    torch.save(output, f'{args.data_path}/activation.{style}.train.bloom-7b')
