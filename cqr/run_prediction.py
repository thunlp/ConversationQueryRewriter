
import argparse
import json
import logging
import random
import torch

from tqdm import tqdm, trange

from cqr.inference_model import InferenceModel
from cqr.utils import NUM_FOLD, set_seed

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument('--input_file', type=str, required=True, 
                        help="Input json file for predictions. Do not add fold suffix when cross validate, i.e. use 'data/eval_topics.jsonl' instead of 'data/eval_topics.jsonl.0'")
    parser.add_argument('--output_file', type=str, required=True,
                        help="Output json file for predictions")
    parser.add_argument("--cross_validate", action='store_true',
                        help="Set when doing cross validation")

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

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)

    MAX_LENGTH = 100
    if args.length < 0:
        args.length = MAX_LENGTH  # avoid infinite loop

    if not args.cross_validate:
        inference_model = InferenceModel(args)
        with open(args.input_file , 'r') as fin, open(args.output_file, 'w') as fout:
            for line in tqdm(fin, desc="Predict"):
                record = json.loads(line)
                prediction = inference_model.predict(record['input'])
                record['output'] = prediction
                fout.write(json.dumps(record) + '\n')
    else:
        # K-Fold Cross Validation
        model_path = args.model_path
        with open(args.output_file, 'w') as fout:
            for i in range(NUM_FOLD):
                logger.info("Predict Fold #{}".format(i))
                args.model_path = "%s-%d" % (model_path, i)
                inference_model = InferenceModel(args)
                input_file = "%s.%d" % (args.input_file, i)
                with open(input_file , 'r') as fin:
                    for line in tqdm(fin, desc="Predict"):
                        record = json.loads(line)
                        prediction = inference_model.predict(record['input'])
                        record['output'] = prediction
                        fout.write(json.dumps(record) + '\n')
    logger.info("Prediction saved to %s", args.output_file)


if __name__ == '__main__':
    main()

