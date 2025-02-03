""" 

    gpttransitions.py 
    
    Dated: 24.02.2024
    Handy tool I built for myself to analyse GPT2's logit scores. 

"""


from transformers import GPT2Tokenizer, AutoModelForCausalLM

class Analyst:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.base = self.tokenizer.get_vocab()

    def washer(self, user_input):
        return self.tokenizer.encode(user_input, output_attentions=True, return_tensors="pt")
        
    def process_response(self, response_text):
        inputs = self.tokenizer([response_text], return_tensors="pt")
        #outputs = self.model.generate(
            # **inputs, max_new_tokens=50, 
            # return_dict_in_generate=True, 
            # output_scores=True, 
            # output_hidden_states=True, 
            # output_attentions=True)
        outputs = self.model.generate(**inputs, 
                        max_new_tokens=50, 
                        return_dict_in_generate=True, 
                        output_scores=True, 
                        output_hidden_states=True,
                        output_attentions=True,
                        #top_k=9,
                        top_p=0.67,
                        temperature=1.1, 
                        do_sample=True,
                        repetition_penalty=1.1,
                    )
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs.sequences[:, input_length:]

        j = 0 
        scores = []
        for tok, score in zip(generated_tokens[0], transition_scores[0]):
            # | token | token string | logits | probability
            print(f"| {tok:5d} | {self.tokenizer.decode(tok):8s} | {score.numpy():.4f} | {torch.exp(score):.2%}")
            j += 1
            
            scores.append({
                'token': tok.item(),
                'word': self.tokenizer.decode(tok),
                'transitionScore': score.numpy(),
                'prob': torch.exp(score).item(),
                'count': input_length + j, # from the input response to the fullstop
                'full': self.tokenizer.batch_decode(outputs.sequences[:, :input_length+j], skip_special_tokens=True)[0],
                'given': response_text,
            })
            if tok in ['.']:
                print(f'{input_length}:{j + input_length}')
        
        hidden = outputs.hidden_states[input_length:input_length+j+1]

        return scores, hidden, outputs
