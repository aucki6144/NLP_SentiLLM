# -*- coding:utf-8 -*-　
# Last modify: Liu Wentao
# Description: Skeleton for fine-tuning with SemEval data on track A
# Note:

import torch
from transformers import pipeline

model_id = "./home/checkpoints_hf/Llama-3.2-1B-hf"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print(pipe("The key to life is"))