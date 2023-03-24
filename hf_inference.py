from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse    

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prompt", help="input prompt as string", default="Text-based generative AI has so much potential because")
parser.add_argument("-l", "--length_max", help="maximum length as integer", default=100, type=int)
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained("/workspace/Port_FasterTransformer/build/model/GPT-NeoXT-20B-hf-v0", torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True) # low_cpu_mem_usage=False
tokenizer = AutoTokenizer.from_pretrained("/workspace/Port_FasterTransformer/build/model/GPT-NeoXT-20B-hf-v0")

tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

input_ids = tokenizer(args.prompt, return_tensors="pt", padding=True).input_ids
input_ids = input_ids.to('cuda')

gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=args.length_max)
gen_text = tokenizer.batch_decode(gen_tokens)[0]

# output
print(f"\nPrompt: {args.prompt}\n")
print(f"Output: {gen_text}")
