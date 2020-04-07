
import argparse
import json
import numpy as np
import random
import torch

from inference_model import InferenceModel

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument('--input_file', type=str, required=True, 
                        help="Input json file for predictions")
    parser.add_argument('--output_file', type=str, required=True,
                        help="Output json file for predictions")
    parser.add_argument("--length", type=int, default=20,
                        help="Maximum length of output sequence")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="temperature of 0 implies greedy sampling")
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    set_seed(args)

    MAX_LENGTH = 100
    if args.length < 0:
        args.length = MAX_LENGTH  # avoid infinite loop

    inference_model = InferenceModel(args)

    with open(args.input_file , 'r') as fin, open(args.output_file, 'w') as fout:
        for line in fin:
            record = json.loads(line)
            prediction = inference_model.predict(record['input'])
            record['output'] = prediction
            fout.write(json.dumps(record) + '\n')

if __name__ == '__main__':
    main()

