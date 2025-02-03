""" 

    local_point.py 

    Contains my initial model handler for inferencing logit scores of the local/offline stored GPT2. 
    This script remains here as a self reminder to never code like this again. 
    
"""

import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

search_params = {
    "top_k": 90,
    "top_p": 0.90,  # Corrected duplicate key
    "temperature": 1.1,  # Corrected syntax
    "do_sample": True,
    "repetition_penalty": 1.1,
}

output_params = {
    'output_scores': True,
    'output_logits': False,
    'output_attentions': False,
    'output_hidden_states': False,
    'output_past_key_values': False,
}

off_params = {
    "max_new_tokens": 75,
    "num_return_sequences": 1
}

class KittyBasement:
    def __init__(self, **kwargs):
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.generation_config.update(**kwargs)
        
    def __call__(self, user_input):
        tokens = self.tokenizer(user_input, return_attention_mask=True, return_tensors="pt")
        output = self.model.generate(**tokens)
        
        self.ids = tokens.data["input_ids"].squeeze(0) 
        self.mask = tokens.data["attention_mask"]
        # [num_input], [1, num_input]
        try:
            print(output)
            self.full_ids = output["sequences"].squeeze(0) # [1, num_input+num_gen]
            self.full_text = KittyBasement.tokenizer.decode(self.full_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            # [num_input+num_gen], [num_input+num_gen]
            
            self.pliers_ids = self.full_ids[self.ids.shape[-1]:]
            self.pliers_text = KittyBasement.tokenizer.decode(self.pliers_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            self.scores = torch.cat(output["scores"]) # note: only gives for the pliers
            self.emit = KittyBasement.model.compute_transition_scores(output["sequences"], output["scores"], normalize_logits=True)
            # [16, 50257] (== [max_new_tokens, vocab_size] == [num_gen, vocab_size]), [1, num_gen]
            print(self.full_text)
        except ValueError as e:
            print(e)
        
if __name__ == '__main__':
    kitty = KittyBasement(**output_params, **off_params)
    kitty('hey how are you?')
