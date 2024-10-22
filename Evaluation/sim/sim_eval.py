import argparse, os, json

from content_similarity import wieting_sim


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
                    pred_list.append(json_dict['output'])
            
            # calculate the similarity
            similarity_by_sent = wieting_sim(args, input_list, pred_list)
            avg_sim_by_sent = similarity_by_sent.mean()
            print(avg_sim_by_sent)
            
            with open(os.path.join(args.out_path, f"predict-from-{style_name}.txt"), 'w', encoding='utf-8') as wf:
                for score in similarity_by_sent:
                    wf.write(str(score) + '\n')
                wf.write(f'average similarity (transfer from {style_name}) score: ' + str(avg_sim_by_sent))

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