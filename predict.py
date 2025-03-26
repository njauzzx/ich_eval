import os
import json
import torch
from vllm import LLM,SamplingParams
from vllm.lora.request import LoRARequest
import argparse

template_map = {
    "default":"{prompt}\n输出：",
    "qwen":"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
<|im_start|>user\n{prompt}<|im_end|>\n\
<|im_start|>assistant\n",
    "baichuan2":"<reserved_106>\n{prompt}<reserved_107>",
    "chatglm3":"[gMASK]sop<|user|>\n{prompt}<|assistant|>\n",
    "glm4":"[gMASK] <sop> <|user|> \n{prompt}<|assistant|>\n",
    "intern2":"<s><|im_start|> system\n\
You are an AI assistant whose name is InternLM (书生·浦语).\n\
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n\
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.<|im_end|> \n\
<|im_start|> user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
    "llama2":"<s> [INST] <<SYS>>\n\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n\
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n\
<</SYS>>\n\n{prompt}[/INST]",
    "llama3":"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\
You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n\
{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "mistral":"<s>[INST]{prompt}[/INST] ",
    "falcon":"User:{prompt}\nFalcon:",
    "gemma":"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n",
    "qwen_think": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\nthink\n",
    "s1":"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\nthink",
    "deepseek-qwen":"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}\nPlease reason step by step, and put your final answer within boxed/{{/}}<|im_end|>\n<|im_start|>assistant\nthink\n",
    "deepseek-qwen":""
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',type=str,default=None)
    parser.add_argument('--template',type=str,default=None)
    parser.add_argument('--data_path',type=str,default=None)
    parser.add_argument('--output_path',type=str,default=None)
    parser.add_argument('--lora_path',type=str,default=None)
    parser.add_argument('--max_sample',type=int,default=None)
    parser.add_argument('--gpu_num',type=int,default=None)
    parser.add_argument('--max_len',type=int,default=None)
    parser.add_argument('--gpu_memory_utilization',type=float,default=None)
    parser.add_argument('--max_tokens',type=int,default=False)
    args = parser.parse_args()
    
    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.95, repetition_penalty=1)
    if args.lora_path is not None:
        llm = LLM(model=args.model_path,tensor_parallel_size=args.gpu_num, trust_remote_code=True,max_model_len=args.max_len,gpu_memory_utilization=args.gpu_memory_utilization,dtype=torch.float16,enable_lora=True, max_seq_len_to_capture=args.max_len)
    else:
        llm = LLM(model=args.model_path,tensor_parallel_size=args.gpu_num, trust_remote_code=True,max_model_len=args.max_len,gpu_memory_utilization=args.gpu_memory_utilization,dtype=torch.float16,max_seq_len_to_capture=args.max_len)

    prompts = []
    predicts = []
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    test_data = json.load(open(args.data_path,'r',encoding='utf'))
    if args.max_sample:
        test_data = test_data[:args.max_sample]
    for line in test_data:
        prompts.append(template_map[args.template].format(prompt=f"{line['instruction']}{line['input']}"))
    if args.lora_path is not None:
        outputs = llm.generate(prompts,sampling_params,lora_request=LoRARequest("lora", 1, lora_path=args.lora_path))
    else:
        outputs = llm.generate(prompts,sampling_params)
    for i in range(len(outputs)):
        generated_text = outputs[i].outputs[0].text.strip(' ')
        predicts.append({'instruction':test_data[i]['instruction'],'input':test_data[i]['input'],'ref':test_data[i]['output'],'pre':generated_text})
    with open(os.path.join(args.output_path,'predict.json'), "w", encoding='utf-8') as writer:
        json.dump(predicts,writer,ensure_ascii=False,indent=2)

if __name__ == "__main__":
    main()