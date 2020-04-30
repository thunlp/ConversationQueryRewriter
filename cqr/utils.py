
import random
import torch
import numpy as np

NUM_FOLD = 5
QUESTION_WORD_LIST = ["what", "when", "why", "who", "how", "where", "whose", "is", "are", "were", "was", "do", "does", "did", "can"]
OTHER_WORD_LIST = ["tell"]
special_tokens_dict = {'sep_token': '<SEP>', 'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>'}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

