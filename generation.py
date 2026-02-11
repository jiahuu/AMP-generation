import sys
sys.path.append('../')
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM
import torch
from transformers import AutoTokenizer
import numpy as np
from utils.set_seed import set_seed
from tqdm import tqdm
import os
set_seed(42)

project_dir_path = os.getcwd()

device = "cuda:3"
peft_model_id = f"soft_prompt_PROMPT_TUNING_CAUSAL_LM_E200_LR0.0001_BS16_ML128_VT20"
peft_model_path = project_dir_path + f"/cache/soft/{peft_model_id}"


config = PeftConfig.from_pretrained(peft_model_path)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, peft_model_path)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path) 
print(model)
model.to(device)
model.eval()

input_p = ['<|endoftext|>']
inputs = tokenizer(input_p, return_tensors="pt")

seq_list = list()

with torch.no_grad():
    gen_seq_num = 1000 
    batch_size= 50
    seq_list = list()
    for i in tqdm(range(int(gen_seq_num/batch_size))):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=8, do_sample=True, top_k=500, repetition_penalty=1.2, num_return_sequences=batch_size, eos_token_id=0)
        seq_res = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
        seq_list += seq_res

    remain_seq_num = gen_seq_num-int(gen_seq_num/batch_size)*batch_size
    if remain_seq_num > 0:
        outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=8, do_sample=True, top_k=500, repetition_penalty=1.2, num_return_sequences=remain_seq_num, eos_token_id=0)
        seq_list += seq_res

    np.save(project_dir_path + f'/designed_amp_mxlen30.npy', seq_list)