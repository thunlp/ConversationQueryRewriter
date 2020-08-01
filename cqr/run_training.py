
import argparse
import logging
import json
import os
import torch

from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm, trange
from transformers import  GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup

from cqr.dataset import QueryRewriteDataset
from cqr.utils import NUM_FOLD, set_seed, special_tokens_dict

logger = logging.getLogger(__name__)


def collate_fn(batch_dataset: list):
    return_tuple = [[], [], [], [], []]
    for example in batch_dataset:
        return_tuple[0].append(example.topic_number)
        return_tuple[1].append(example.query_number)
        return_tuple[2].append(example.ids)
        return_tuple[3].append(example.labels)
        return_tuple[4].append(example.pred_begin_pos)
    return_tuple[2] = torch.tensor(return_tuple[2])
    return_tuple[3] = torch.tensor(return_tuple[3])
    return_tuple = tuple(return_tuple)
    return return_tuple


def train(args, train_dataset, model, tokenizer, logger, cross_validate_id=-1):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            inputs, labels = (batch[2], batch[3])  # get ids and labels
            inputs = inputs.to(args.device)  # batch_size * block_size
            labels = labels.to(args.device)
            model.train()
            outputs = model(inputs, labels=labels)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            del inputs
            del outputs
            torch.cuda.empty_cache()

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            del loss
            torch.cuda.empty_cache()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = 'checkpoint'
                    output_dir = args.output_dir + (('-' + str(cross_validate_id)) if cross_validate_id != -1 else "")
                    # Save model checkpoint
                    output_dir = os.path.join(output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_name_or_path", default="gpt2-medium", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--block_size", default=150, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="Path of training file. Do not add fold suffix when cross validate, i.e. use 'data/eval_topics.jsonl' instead of 'data/eval_topics.jsonl.0'")
    parser.add_argument("--cross_validate", action='store_true',
                        help="Set when doing cross validation")
    parser.add_argument("--init_from_multiple_models", action='store_true',
                        help="Set when initialize from different models during cross validation (Model-based+CV)")

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args)

    config_class, model_class, tokenizer_class = GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

    if not args.cross_validate:
        config = config_class.from_pretrained(args.model_name_or_path)
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
        tokenizer.add_special_tokens(special_tokens_dict)
        model = model_class.from_pretrained(args.model_name_or_path)
        model.resize_token_embeddings(len(tokenizer))  # resize
        model.to(args.device)
	
        if args.block_size <= 0:
            args.block_size = tokenizer.max_len_single_sentence
        args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

        # Training
        logger.info("Training/evaluation parameters %s", args)
        train_dataset = QueryRewriteDataset([args.train_file], tokenizer, args)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, logger)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Saving
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    else:
        # K-Fold Cross Validation
        for i in range(NUM_FOLD):
            logger.info("Training Fold #{}".format(i))
            suffix = ('-' + str(i)) if args.init_from_multiple_models else ''
            config = config_class.from_pretrained(args.model_name_or_path + suffix)
            tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path + suffix)
            tokenizer.add_special_tokens(special_tokens_dict)
            model = model_class.from_pretrained(args.model_name_or_path + suffix)
            model.resize_token_embeddings(len(tokenizer))  # resize
            model.to(args.device)

            if args.block_size <= 0:
                args.block_size = tokenizer.max_len_single_sentence
            args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    
            logger.info("Training/evaluation parameters %s", args)
            train_files = ["%s.%d" % (args.train_file, j) for j in range(NUM_FOLD) if j != i]
            logger.info("train_files: {}".format(train_files))
            train_dataset = QueryRewriteDataset(train_files, tokenizer, args)
            global_step, tr_loss = train(args, train_dataset, model, tokenizer, logger, cross_validate_id=i)
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

            # Create output directory if needed
            output_dir = args.output_dir + '-' + str(i)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            logger.info("Saving model checkpoint to %s", output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))

            del model
            torch.cuda.empty_cache() 


if __name__ == "__main__":
    main()

