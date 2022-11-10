import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Dict
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from utils.constants import NEGATIVE_INF
from utils.utils import logits_to_entropy, mask_pad


class Policy:
    def __init__(self, model_name, temperature, device, reward_cond=False, tree_tokens=None):
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.device = device

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, pad_token="<|endoftext|>")
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        if reward_cond:
            self.tokenizer.add_tokens(tree_tokens, special_tokens=True)

            weights = self.model.get_input_embeddings().weight.detach().numpy()
            mean_weights, std_weights = np.mean(weights, axis=0), np.std(weights, axis=0)
            new_inits = np.vstack([np.random.normal(loc=mean_weights, scale=std_weights) for _ in tree_tokens])

            self.model.resize_token_embeddings(len(self.tokenizer))
            with torch.no_grad():
                new_inits = torch.tensor(new_inits)
                self.model.get_input_embeddings().weight[-len(tree_tokens):, :] = new_inits

        self.model = self.model.to(self.device)
        self.model.parallelize()

        self.temperature = temperature

    def sample(self,
               prompts: Union[str, List[str]] = None,
               input_ids: torch.Tensor = None,
               attention_mask: torch.Tensor = None,
               max_len: int = 20,
               min_len: int = 3,
               sample: bool = True,
               top_k: int = None,
               top_p: float = None,
               temperature: float = None) -> Dict[str, Union[torch.Tensor, List[str]]]:

        if temperature is None:
            temperature = self.temperature

        if prompts is not None:
            assert input_ids is None and attention_mask is None, 'repeated input'
            if isinstance(prompts, str):
                prompts = [prompts]

            encodings_dict = self.tokenizer(prompts, return_tensors="pt", padding=True)
            input_ids = encodings_dict['input_ids'].to(self.device)
            attention_mask = encodings_dict['attention_mask'].to(self.device)

        else:
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

        model_kwargs = {'attention_mask': attention_mask}
        batch_size, input_seq_len = input_ids.shape

        logits_warper = self.model._get_logits_warper(
            top_k=top_k, top_p=top_p, temperature=temperature, num_beams=1
        )

        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=self.device)
        output_logprob = torch.zeros([batch_size, 0], dtype=torch.float, device=self.device)
        output_mask = torch.ones([batch_size, 0], dtype=torch.long, device=self.device)

        self.model.eval()
        with torch.no_grad():
            for step in range(max_len):

                # prepare model inputs
                model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)

                # forward pass to get next token
                outputs = self.model(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )

                # in the first decoding step, we want to use the 'real' last position for each sentence
                if step == 0:
                    last_non_masked_idx = torch.sum(attention_mask, dim=1) - 1
                    next_token_logits = outputs.logits[range(batch_size), last_non_masked_idx, :]
                else:
                    next_token_logits = outputs.logits[:, -1, :]

                if step < min_len:
                    next_token_logits[:, self.model.config.eos_token_id] = float('-inf')
                log_prob = F.log_softmax(next_token_logits, dim=-1)

                if sample:
                    # Temperature (higher temperature => more likely to sample low probability tokens)
                    next_token_scores = logits_warper(input_ids, next_token_logits)
                    probs = F.softmax(next_token_scores, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    # Greedy decoding
                    next_tokens = torch.argmax(next_token_logits, dim=-1)

                # finished sentences should have their next token be a padding token
                next_tokens = next_tokens * unfinished_sequences + self.tokenizer.pad_token_id * (1 - unfinished_sequences)

                    # update output mask
                output_mask = torch.cat([output_mask, unfinished_sequences[:, None]], dim=-1)
                # update output log probability
                token_logprob = torch.gather(log_prob, 1, next_tokens[:, None]).squeeze(1)
                token_logprob = token_logprob * unfinished_sequences + NEGATIVE_INF * (1 - unfinished_sequences)
                output_logprob = torch.cat([output_logprob, token_logprob[:, None]], dim=-1)

                # update generated ids, model inputs for next step
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                model_kwargs = self.model._update_model_kwargs_for_generation(
                    outputs, model_kwargs, is_encoder_decoder=self.model.config.is_encoder_decoder
                )

                # if eos_token was found in one sentence, set sentence to finished
                unfinished_sequences = unfinished_sequences.mul((next_tokens != self.tokenizer.eos_token_id).long())

                if unfinished_sequences.max() == 0:
                    break

        response_ids = input_ids[:, input_seq_len:]
        response_text = [self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                         for output in response_ids]

        prompt_ids = input_ids[:, :input_seq_len]
        if prompts is None:
            prompts = [self.tokenizer.decode(query, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                       for query in prompt_ids]

        return {
            'query/input_ids': prompt_ids,
            'query/text': prompts,
            'query/mask': attention_mask,
            'response/input_ids': response_ids,
            'response/text': response_text,
            'response/mask': output_mask,
            'response/log_prob': output_logprob,
        }

    def forward_pass(self,
                     query_input_ids: torch.Tensor,
                     query_mask: torch.Tensor,
                     response_input_ids: torch.Tensor,
                     response_mask: torch.Tensor):

        query_input_ids = query_input_ids.to(self.device)
        query_mask = query_mask.to(self.device)
        response_input_ids = response_input_ids.to(self.device)
        response_mask = response_mask.to(self.device)

        batch_size, query_seq_len = query_input_ids.shape
        input_ids = torch.cat([query_input_ids, response_input_ids], dim=-1)
        model_kwargs = {'attention_mask': torch.cat([query_mask, response_mask], dim=-1)}
        model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass to get next token
        outputs = self.model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        # get the first logit
        query_logits = outputs.logits[:, :query_seq_len, :]
        last_non_masked_idx = torch.sum(query_mask, dim=1) - 1
        first_logits = query_logits[range(batch_size), last_non_masked_idx, :]
        # get the second to last logit
        response_logits = outputs.logits[:, query_seq_len:-1, :]
        logits = torch.cat([first_logits[:, None], response_logits], dim=1)

        log_prob = F.log_softmax(logits, dim=-1)
        output_logprob = torch.gather(log_prob, 2, response_input_ids[:, :, None]).squeeze(2)
        output_entropy = logits_to_entropy(logits)
        lm_loss = -1. * output_logprob

        return {
            'response/log_prob': mask_pad(output_logprob, response_mask),
            'response/lm_loss': mask_pad(lm_loss, response_mask),
            'response/entropy': mask_pad(output_entropy, response_mask),
            'response/logits': logits,
        }
