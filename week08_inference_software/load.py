from transformers import AutoModelForCausalLM
import torch

device = 'cpu'
# model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B', torch_dtype=torch.float16).to(device)
large_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.float16).to(device)