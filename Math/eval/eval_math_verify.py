# import debugpy; debugpy.connect(('100.96.118.44', 5685))
import pandas as pd
import os
import json
from math_verify import parse, verify
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='qwen3-8b-base-rpe')
parser.add_argument('--test_file', type=str, default='aime-2024-deepscaler-repeat128.parquet')
parser.add_argument('--temperature', type=float, default=0.6)
parser.add_argument('--is_olympiad', default=False, action='store_true')
args = parser.parse_args()
print(args)

def verify_task(ans, ground, verbose=False):
    parsed_ground = parse(ground)
    parsed_ans = parse(ans)
    verified_correct = verify(parsed_ground, parsed_ans)
    if verbose:
        print(f"====\nparsed_GT: {parsed_ground}, parsed_predicted: {parsed_ans}, verified_correct: {verified_correct}\n====\n")
    return verified_correct

if args.temperature != 0.6:
    read_file = f'generation_results/eval_{args.test_file.split(".")[0]}_{args.model}_t{args.temperature}.parquet'
else:
    read_file = f'generation_results/eval_{args.test_file.split(".")[0]}_{args.model}.parquet'

df = pd.read_parquet(read_file)
print(f"read generation result file: {read_file}")


if args.temperature != 0.6:
    save_dir = f'eval_outputs/{args.test_file.split(".")[0]}_{args.model}_t{args.temperature}'
else:
    save_dir = f'eval_outputs/{args.test_file.split(".")[0]}_{args.model}'
os.makedirs(save_dir, exist_ok=True)

correct_count = 0
total_count = 0
format_count = 0

for idx, row in tqdm(df[['prompt', 'output', 'reward_model']].iterrows(), total=len(df), desc='evaluating...'):
    prompt = row['prompt'][0]
    output = row['output']
    gt = row['reward_model']['ground_truth']
    answer = gt if type(gt) == str else str(gt)

    total_count += 1
    if '\\boxed{' in output:
        format_count += 1

    # verify
    if args.is_olympiad:
        print('>>>olympiad answer:', answer)
        answer = eval(answer)
        assert type(answer) == list and len(answer) == 1
        answer = '\\boxed{' + answer[0] + '}'
    else:
        answer = '\\boxed{' + answer + '}'
    verified_correct = verify_task(output, answer, verbose=False)
    
    if verified_correct:
        correct_count += 1
    
    # save to json file
    data = {
        'prompt': prompt,
        'output': output,
        'reward_model': row['reward_model'] if not args.is_olympiad else {'ground_truth': answer, 'style': 'rule'},
        'verified_correct': verified_correct
    }
    with open(f'{save_dir}/row_{idx}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# calculate and print accuracy
accuracy = correct_count / total_count if total_count > 0 else 0
print(f"total samples: {total_count}")
print(f"correct samples: {correct_count}")
print(f"accuracy: {accuracy:.2%}")
print(f"number of answers with \\boxed instruction following: {format_count}, ratio: {format_count/total_count:.2%}")
print(f"saved to: {save_dir}")
