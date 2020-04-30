
import torch
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def top_p_filtering(logits, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


class InferenceModel:

    def __init__(self, args):
        model_class, tokenizer_class = GPT2LMHeadModel, GPT2Tokenizer
        self.tokenizer = tokenizer_class.from_pretrained(args.model_path)
        self.model = model_class.from_pretrained(args.model_path)
        self.model.to(args.device)
        self.model.eval()

        self.device = args.device
        self.length = args.length
        if self.model.config.max_position_embeddings < args.length:
            self.length = model.config.max_position_embeddings # No generation bigger than model size 
        self.temperature = args.temperature
        self.top_p = args.top_p

        self.special_tokens = ['<SEP>', '<PAD>', '<BOS>', '<EOS>']

    def get_input_seq(self, input_sents):
        inputs = []
        for sent in input_sents:
            inputs.extend(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sent)))
            inputs.append(self.tokenizer.sep_token_id)
        inputs.pop()
        inputs.append(self.tokenizer.bos_token_id)
        return inputs

    def remove_special_tokens(self, text):
        # Remove special tokens from the output text in rare cases
        for token in self.special_tokens:
            text = text.replace(token, "")
        return text

    def predict(self, input_sents):
        input_ids = self.get_input_seq(input_sents)
        input_length = len(input_ids)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            for _ in range(self.length):
                inputs = {'input_ids': input_ids}
    
                outputs = self.model(**inputs)
                next_token_logits = outputs[0][:, -1, :] / (self.temperature if self.temperature > 0 else 1.)
    
                filtered_logits = top_p_filtering(next_token_logits, top_p=self.top_p)
                if self.temperature == 0: # greedy sampling:
                    next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
                else:
                    next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                new_token = to_list(next_token)
                if self.tokenizer.decode(new_token[0]).strip() == "<EOS>":
                    break
                input_ids = torch.cat((input_ids, next_token), dim=1)

        pred_ids = to_list(input_ids[0, input_length:])
        pred_text = self.tokenizer.decode(pred_ids, clean_up_tokenization_spaces=True)
        pred_text = self.remove_special_tokens(pred_text)
        
        return pred_text 

