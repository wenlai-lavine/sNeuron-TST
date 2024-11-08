import argparse
import torch
from tqdm import tqdm
import os

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
    
    ## 
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)
    
    # total_style_neurons_list
    total_neurons_list = []
    
    statistic_out_file = open(os.path.join(args.out_path, 'statistic_neurons.csv'), 'w')
    statistic_out_file.write('Style, Style_Name, Neurons\n')
    
    for style in tqdm(style_list):
        positive_style = style_dict[style][0]
        negative_style = style_dict[style][1]
        
        ## activation
        positive_activation = torch.load(f"{args.activation_path}/activation.{style}.{positive_style}.train.llama-7b")
        negative_activation = torch.load(f"{args.activation_path}/activation.{style}.{negative_style}.train.llama-7b")
        
        num_layers, intermediate_size = positive_activation['over_zero'].size()
        
        ## 1. positive neurons
        positive_negative_difference = positive_activation['over_zero'] - negative_activation['over_zero']
        positive_negative_difference_sorted = torch.topk(positive_negative_difference.view(-1), args.threshold)
        positive_neurons = positive_negative_difference_sorted.indices.tolist()
        
        final_positive_neurons_list = [[] for _ in range(num_layers)]
        for pn in positive_neurons:
            row_index = pn // intermediate_size
            col_index = pn % intermediate_size
            final_positive_neurons_list[row_index].append(col_index)
        
        final_positive_neurons = []
        for fpn in final_positive_neurons_list:
            if fpn:
                tmp_list = sorted(fpn)
                final_positive_neurons.append(torch.tensor(tmp_list).long())
            else:
                final_positive_neurons.append(torch.tensor(fpn).long())
        
        # torch.save(final_positive_neurons, f"{args.out_path}/{style}.{positive_style}.llama-7b")
        
        total_neurons_list.append(final_positive_neurons)
        
        statistic_out_file.write(f"{style}, {positive_style}, {str(len(positive_neurons))}\n")
        
        ## 2. negative neurons
        negative_positive_difference = negative_activation['over_zero'] - positive_activation['over_zero']
        negative_positive_difference_sorted = torch.topk(negative_positive_difference.view(-1), args.threshold)
        negative_neurons = negative_positive_difference_sorted.indices.tolist()
        
        final_negative_neurons_list = [[] for _ in range(num_layers)]
        for nn in negative_neurons:
            row_index = nn // intermediate_size
            col_index = nn % intermediate_size
            final_negative_neurons_list[row_index].append(col_index)
        
        final_negative_neurons = []
        for fnn in final_negative_neurons_list:
            if fnn:
                tmp_list = sorted(fnn)
                final_negative_neurons.append(torch.tensor(tmp_list).long())
            else:
                final_negative_neurons.append(torch.tensor(fnn).long())
        
        # torch.save(final_negative_neurons, f"{args.out_path}/{style}.{negative_style}.llama-7b")
        
        statistic_out_file.write(f"{style}, {negative_style}, {str(len(negative_neurons))}\n")
        
        total_neurons_list.append(final_negative_neurons)
        
        torch.save(total_neurons_list, f"{args.out_path}/{style}.llama-7b")
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threshold", type=int, default=10000)
    parser.add_argument("-ap", "--activation_path", type=str, default="formal")
    parser.add_argument("-tap", "--task_activation_path", type=str, default="formal")
    parser.add_argument("-o", "--out_path", type=str, default="")
    args = parser.parse_args()
    main(args)