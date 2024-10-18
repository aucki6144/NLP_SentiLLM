# -*- coding:utf-8 -*-　
# Last modify: Liu Wentao
# Description:
# Note:
import os
import string

PROMPT_PATH = "./home/template/prompt_template.txt"

def get_prompt_template():

    prompt_template = ""
    with open(PROMPT_PATH) as f:
        prompt_template = f.read()

    template = string.Template(prompt_template)
    return template

def generate_prompt():
    return get_prompt_template()