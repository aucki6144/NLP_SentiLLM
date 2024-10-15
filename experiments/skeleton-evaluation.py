import argparse
import transformers

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def main(args):

    # TODO: Evaluation skeleton

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_path',
        '-m',
        type=str,
        required=True,
        help='Path to fine-tuned model',
    )

    parser.add_argument(
        '--data_set',
        '-d',
        type=str,
        required=False,
        default='emotion',
        help='Path to evaluation dataset',
    )

    config_args = parser.parse_args()

    main(config_args)