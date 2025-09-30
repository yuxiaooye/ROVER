# import debugpy; debugpy.connect(('100.98.4.207', 5680))

# from openai import AsyncOpenAI
import pandas as pd
import time
import asyncio
import tqdm
from tqdm.contrib.concurrent import process_map
# from math_dapo import compute_score
import argparse
import os
import openai
import json

DEBUG = False

parser = argparse.ArgumentParser()
parser.add_argument('--temperature', type=float, default=0.6)
parser.add_argument('--top_p', type=float, default=0.95)
parser.add_argument('--max_tokens', type=int, default=20480)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--model', type=str, default='qwen3-8b-base-rpe')
parser.add_argument('--test_file', type=str, default='aime-2024-deepscaler-repeat128.parquet')
parser.add_argument('--port', type=int, default=8080, help='vLLM port')

args = parser.parse_args()

# set vllm port compativle with OpenAI
openai.api_key = "token-abc123"
openai.api_base = f"http://localhost:{args.port}/v1"

async def chat(messages, temperature, top_p, max_tokens, model):
    try:
        completion = await openai.ChatCompletion.acreate(
            model=model,
            messages=messages.tolist(),
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=1,
        )
        return completion, True
    except Exception as e:
        print(f"eval error: {str(e)}")
        return None, False

async def process_batch(batch_tasks):
    try:
        responses = await asyncio.gather(*[chat(*task) for task in batch_tasks])
        return responses
    except Exception as e:
        print(f"process batch error: {str(e)}")
        return None

async def eval(all_tasks, batch_size):
    batch_size = batch_size if not DEBUG else 1
    start = time.time()
    results = []
    
    with tqdm.tqdm(total=len(all_tasks)) as pbar:
        for i in range(0, len(all_tasks), batch_size):
            batch = all_tasks[i:i + batch_size]
            batch_results = await process_batch(batch)
            if batch_results:
                results.extend(batch_results)
            pbar.update(len(batch))
    
    end = time.time()
    total_time = end - start
    print(f"total time: {total_time} seconds")
    return results, total_time

def verify(arg):
    rsp, reward_model = arg
    # return compute_score(rsp, reward_model['ground_truth'])
    return {'acc': 0.0, 'score': 0.0}

def main(args):
    df = pd.read_parquet(args.test_file)
    if DEBUG:
        df = df.head(1)
    # prepare tasks
    tasks = [(msg, args.temperature, args.top_p, args.max_tokens, args.model) 
             for msg in df['prompt']]  
    
    ret, total_time = asyncio.run(eval(tasks, args.batch_size))
    
    # count skipped questions
    skipped_count = 0
    outputs = []
    for r, success in ret:
        if success:
            outputs.append(r.choices[0].message.content)
        else:
            skipped_count += 1
            outputs.append("")  # for failed cases, output is empty string
    df['output'] = outputs
    
    # 
    # save_dir = f'eval_outputs/eval_outputs_{args.model}'
    # os.makedirs(save_dir, exist_ok=True)
    # for idx, row in df[['output', 'reward_model']].iterrows(): 
    #     data = {
    #         'output': row['output'],
    #         'reward_model': row['reward_model']
    #     }
    #     with open(f'{save_dir}/row_{idx}.json', 'w', encoding='utf-8') as f:
    #         json.dump(data, f, ensure_ascii=False, indent=2)

    df['res'] = process_map(verify, df[['output', 'reward_model']].values, max_workers=50, chunksize=1)
    timestamp = time.strftime("%m%d_%H%M", time.localtime())
    
    # build save file name
    if args.temperature != 0.6:
        base_filename = f'eval_{os.path.basename(args.test_file).split(".")[0]}_{args.model}_t{args.temperature}'
    else:
        base_filename = f'eval_{os.path.basename(args.test_file).split(".")[0]}_{args.model}'
    
    # save generation results
    save_file = f'generation_results/{base_filename}.parquet'
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    df.to_parquet(save_file)
    print(f"model generation saved to {save_file}")
    
    # save time cost info
    time_cost_dir = 'gen_time_cost'
    os.makedirs(time_cost_dir, exist_ok=True)
    time_cost_file = f'{time_cost_dir}/{base_filename}.json'
    
    time_info = {
        'total_time_seconds': total_time,
        'batch_size': args.batch_size,
        'sample_count': len(df),
        'timestamp': timestamp,
        'model': args.model,
        'temperature': args.temperature
    }
    
    with open(time_cost_file, 'w', encoding='utf-8') as f:
        json.dump(time_info, f, ensure_ascii=False, indent=2)
    print(f"time cost info saved to {time_cost_file}")


    
    score = 0
    for i, row in df.iterrows():
        score += row['res']['acc']
    avg_score = score / len(df)
    skip_ratio = skipped_count / len(df)
    
    print(f"acc/mean@32: {avg_score}")
    print(f"skipped ratio: {skip_ratio:.2%} ({skipped_count}/{len(df)})")

if __name__ == "__main__":
    print(args)
    main(args)