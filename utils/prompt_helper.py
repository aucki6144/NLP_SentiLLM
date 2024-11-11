# -*- coding:utf-8 -*-　
# Last modify: Liu Wentao
# Description:
# Note:
import argparse
import os
import string
import json

TEMPLATE_PATH = "./experiments/home/template/template.json"


def get_prompt_label_template(index=0):
    with open(TEMPLATE_PATH, 'r') as file:
        prompt_label_pair = json.load(file)

    if 0 <= index < len(prompt_label_pair):
        pair = prompt_label_pair[index]
    else:
        print("Index out of range for get prompt template")
        return "", ""

    print(f"get_prompt_label_template({index}) using template: {pair}")

    return string.Template(pair['prompt']), string.Template(pair['labels'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--prompt_template',
        '-p',
        type=str,
        required=True,
        help='Prompt template',
    )

    parser.add_argument(
        '--label_template',
        '-l',
        type=str,
        required=True,
    )

    parser.add_argument(
        '--save_dir',
        '-sd',
        type=str,
        required=False,
        default=TEMPLATE_PATH,
        help='Directory to save prompts',
    )

    args = parser.parse_args()

    with open(TEMPLATE_PATH, 'r') as file:
        prompt_label_pair = json.load(file)

    prompt_label_pair.append({
        'prompt': args.prompt_template,
        'labels': args.label_template,
    })

    # Save to a JSON file
    with open(TEMPLATE_PATH, 'w') as f:
        json.dump(prompt_label_pair, f, indent=4)
