## Inferences on Decoding parameters in steering GPT2 sequences from scratch

HF Library tools used: https://huggingface.co/docs/transformers/en/internal/generation_utils

## Adding mask ids to the Tokenizer 




```python
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn 
from transformers import GPT2Tokenizer, GPT2LMHeadModel, LogitsProcessor, LogitsProcessorList, set_seed

DIAL_TOKEN = ["<|user|>","<|agent|>"]
# INTENT_TOKEN = ["<ANSWER>", "<QUESTION>", "<COMMAND>", "<EXPRESSION>"]
# EMOTE_TOKEN = ["UNKNOWN","<NEUTRAL>", '<EMOTION:POS>', '<EMOTION:NEG>']

set_seed(42)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
# pad_token_id == eos_token_id == bos_token_id == <|endoftext|> 
tokenizer.pad_token = tokenizer.eos_token
# tokenizer.add_tokens(DIAL_TOKEN, special_tokens=False)
# model.resize_token_embeddings(len(tokenizer))
tokenizer.add_special_tokens
```




    <bound method SpecialTokensMixin.add_special_tokens of GPT2Tokenizer(name_or_path='gpt2', vocab_size=50257, model_max_length=1024, is_fast=False, padding_side='right', truncation_side='right', special_tokens={'bos_token': '<|endoftext|>', 'eos_token': '<|endoftext|>', 'unk_token': '<|endoftext|>', 'pad_token': '<|endoftext|>'}, clean_up_tokenization_spaces=True),  added_tokens_decoder={
    	50256: AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),
    }>



#### Ensuring the configuration of the tokens are as expected (sometimes you can be calling the wrong model or that the new update may have removed / altered some layers / parameters)


```python
tokenizer.added_tokens_decoder
```




    {50256: AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True)}




```python
tokenizer.added_tokens_encoder
```




    {'<|endoftext|>': 50256}



#### Default generation from gpt2


```python
# define inputs
user_input = "User: I love you, do you love me?.\nAgent:"

# preprocessing the inputs 
inputs = tokenizer(user_input, add_special_tokens=False, return_tensors="pt")
input_length = inputs.input_ids.size(1)
generation_output = model.generate(
    **inputs, 
    # max_length=input_length+50, 
    max_new_tokens=50,
    num_return_sequences=1, 
    no_repeat_ngram_size=4, 
    temperature=1.0, 
    do_sample=True, 
    return_dict_in_generate=True, 
    #logits_processor=custom_logits, 
    output_scores=True
)

# cleans text
print(tokenizer.decode(generation_output.sequences.squeeze(0), skip_special_tokens=False))
```

    Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.


    User: I love you, do you love me?.
    Agent: Ohhh. Not that I know. You need me to talk to a bunch of women. I love you.
    Linda shakes her head. "No," she says before she can finish.
    Bryan grins. "And you're


#### Summary table from the generation config's documentation (Decoder parameters accepted for this library)

| **Arg**                     | **Usage**                                                                                                           | **Type**                           |
|-----------------------------|---------------------------------------------------------------------------------------------------------------------|-----------------------------------|
| `max_length`               | Maximum length of generated tokens (prompt + `max_new_tokens`). Overridden by `max_new_tokens` if set.              | `int` (default: 20)              |
| `max_new_tokens`           | Maximum number of tokens to generate, ignoring the prompt length.                                                   | `int`                            |
| `min_length`               | Minimum length of the sequence to be generated (prompt + `min_new_tokens`). Overridden by `min_new_tokens`.         | `int` (default: 0)               |
| `min_new_tokens`           | Minimum number of tokens to generate, ignoring the prompt length.                                                   | `int`                            |
| `early_stopping`           | Stops beam-based methods early: `True` (stops when `num_beams` candidates are complete), `False` (heuristic), `"never"`. | `bool` or `str` (default: `False`) |
| `max_time`                 | Maximum runtime for generation in seconds. Generation finishes the current pass even if time exceeds.               | `float`                          |
| `stop_strings`             | String(s) that terminate generation if output by the model.                                                         | `str` or `List[str]`             |
| `do_sample`                | Enables multinomial sampling; otherwise uses greedy decoding.                                                       | `bool` (default: `False`)        |
| `num_beams`                | Number of beams for beam search. If `1`, no beam search is performed.                                                | `int` (default: 1)               |
| `num_beam_groups`          | Divides `num_beams` into groups for diverse beam search.                                                            | `int` (default: 1)               |
| `penalty_alpha`            | Balances model confidence and degeneration penalty in contrastive search.                                           | `float`                          |
| `use_cache`                | Whether to use the model's past key/value cache for faster decoding.                                                | `bool` (default: `True`)         |
| `temperature`              | Modulates token probabilities; higher values increase randomness.                                                  | `float` (default: 1.0)           |
| `top_k`                    | Retains only the top-k probability tokens.                                                                          | `int` (default: 50)              |
| `top_p`                    | Retains the smallest set of tokens with cumulative probabilities ≥ `top_p`.                                         | `float` (default: 1.0)           |
| `min_p`                    | Minimum token probability scaled by the most likely token.                                                         | `float`                          |
| `typical_p`                | Retains locally typical tokens with probabilities adding up to `typical_p`.                                         | `float` (default: 1.0)           |
| `epsilon_cutoff`           | Filters tokens with conditional probabilities below `epsilon_cutoff`.                                               | `float` (default: 0.0)           |
| `eta_cutoff`               | Hybrid of typical and epsilon sampling with entropy scaling.                                                        | `float` (default: 0.0)           |
| `diversity_penalty`        | Penalizes beams that generate tokens appearing in other groups during group beam search.                            | `float` (default: 0.0)           |
| `repetition_penalty`       | Penalizes repeated sequences. Higher values reduce repetition.                                                      | `float` (default: 1.0)           |
| `length_penalty`           | Exponential penalty for sequence length in beam-based methods. Longer sequences favored if > 1.0.                   | `float` (default: 1.0)           |
| `no_repeat_ngram_size`     | Ensures n-grams of this size do not repeat during generation.                                                       | `int` (default: 0)               |
| `bad_words_ids`            | Token IDs disallowed in generated sequences.                                                                        | `List[List[int]]`                |
| `forced_bos_token_id`      | Forces the first generated token to be a specific token.                                                            | `int`                            |
| `forced_eos_token_id`      | Forces the last generated token(s) to be specific tokens.                                                           | `int` or `List[int]`             |
| `remove_invalid_values`    | Removes `NaN` or `Inf` values from logits to avoid crashes.                                                         | `bool`                           |
| `num_return_sequences`     | Number of independently generated sequences per input.                                                              | `int` (default: 1)               |
| `output_attentions`        | Returns attention scores if set to `True`.                                                                          | `bool` (default: `False`)        |
| `output_hidden_states`     | Returns hidden states if set to `True`.                                                                             | `bool` (default: `False`)        |
| `output_scores`            | Returns prediction scores if set to `True`.                                                                         | `bool` (default: `False`)        |
| `pad_token_id`             | Token ID for padding.                                                                                               | `int`                            |
| `bos_token_id`             | Token ID for beginning of sequence.                                                                                 | `int`                            |
| `eos_token_id`             | Token ID for end of sequence.                                                                                       | `int` or `List[int]`             |


```python
history = []
user_input = "User: I love you. Do you love me?\nAgent:"
```


```python
class Plankton(LogitsProcessor):
    """
    Logit Scores Custom Altering first attempt. 
    NOTE: This actually iterates token by token NOT sequence by sequence
    """ 
    
    def __init__(self, input_size: int, boost_tokens, anchor_tokens, boost_scale: int, anchor_scale: int = None):
        super().__init__()
        self.input_size = input_size
        self.anchor_tokens = [i for i in anchor_tokens if i != tokenizer.eos_token_id]
        self.boost_tokens = [i for i in boost_tokens if i != tokenizer.eos_token_id]
        self.boost_scale = boost_scale 
        self.anchor_scale = anchor_scale if anchor_scale is not None else self.boost_scale
        self.counter = 0 
        
    def __call__(self, input_ids, logits):
        """This is called very token logits (shape: [1, 50259]). It prints out the scaled logits and decoded token's ids."""

        # logits shape is [1, 50259] ~ [1, vocab_dim]
        logits[:, self.anchor_tokens] -= self.anchor_scale # logits shape being scaled [1, 4] ~ [1, anchor_tokens]
        logits[:, self.boost_tokens] += self.boost_scale # alike anchor, [1, boost_tokens]

        print(f">>>>>>>>>>>>>>>>>>> Function called {self.counter+1}: {tokenizer.decode(input_ids[:, self.input_size:].squeeze(0), skip_special_tokens=False)} \nAnchor Logits: {logits[:, self.anchor_tokens].cpu().numpy()}\nBoost Logits: {logits[:, self.boost_tokens].cpu().numpy()}")
        
        self.counter += 1
        
        return logits
```


```python
# trying out 1 token
custom_logit = Plankton(
    input_size=input_length,
    anchor_tokens=tokenizer.convert_tokens_to_ids(["hate"]),
    boost_tokens=tokenizer.convert_tokens_to_ids(["love", "adore"]),
    boost_scale=10
)

custom_logits = LogitsProcessorList([custom_logit])
inputs = tokenizer(user_input, add_special_tokens=False, return_tensors="pt")
input_length = inputs.input_ids.size(1)
# fixed variables - remains unchanged throughout
generation_output = model.generate(
    **inputs, 
    max_length=input_length+50, 
    num_return_sequences=1, 
    top_k=50,
    top_p=0.7,
    temperature=0.95, 
    stop_strings=["Agent:", "User:", "\n"],
    do_sample=True, 
    return_dict_in_generate=True, 
    tokenizer=tokenizer,
    logits_processor=custom_logits, 
    output_scores=True,
    use_cache=False,
)
output_tokens = tokenizer.decode(generation_output.sequences[:, input_length:].squeeze(0), skip_special_tokens=False)
history.append({
    "inputs": user_input, 
    "outputs": output_tokens,
    **custom_logit.__dict__
})

print(f">>> FULL OUTPUT:\n{output_tokens}")
```

    Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.


    >>>>>>>>>>>>>>>>>>> Function called 1:  
    Anchor Logits: [[-151.03879]]
    Boost Logits: [[-127.9868]]
    >>>>>>>>>>>>>>>>>>> Function called 2:  Yes 
    Anchor Logits: [[-73.75444]]
    Boost Logits: [[-50.006798]]
    >>>>>>>>>>>>>>>>>>> Function called 3:  Yes. 
    Anchor Logits: [[-133.18825]]
    Boost Logits: [[-107.58431]]
    >>>>>>>>>>>>>>>>>>> Function called 4:  Yes. Do 
    Anchor Logits: [[-53.124672]]
    Boost Logits: [[-29.986675]]
    >>>>>>>>>>>>>>>>>>> Function called 5:  Yes. Do you 
    Anchor Logits: [[-135.0587]]
    Boost Logits: [[-112.66008]]
    >>>>>>>>>>>>>>>>>>> Function called 6:  Yes. Do you want 
    Anchor Logits: [[-75.015686]]
    Boost Logits: [[-51.323887]]
    >>>>>>>>>>>>>>>>>>> Function called 7:  Yes. Do you want to 
    Anchor Logits: [[-162.91136]]
    Boost Logits: [[-140.591]]
    >>>>>>>>>>>>>>>>>>> Function called 8:  Yes. Do you want to meet 
    Anchor Logits: [[-101.80152]]
    Boost Logits: [[-78.66078]]
    >>>>>>>>>>>>>>>>>>> Function called 9:  Yes. Do you want to meet me 
    Anchor Logits: [[-131.58554]]
    Boost Logits: [[-107.18489]]
    >>>>>>>>>>>>>>>>>>> Function called 10:  Yes. Do you want to meet me? 
    Anchor Logits: [[-82.53003]]
    Boost Logits: [[-58.055588]]
    >>> FULL OUTPUT:
     Yes. Do you want to meet me?
    


- without logsoftmax or softmax


```python
import pandas as pd 
```


```python
df = pd.DataFrame(history)
```


```python
df["anchor_text"] = tokenizer.batch_decode(df["anchor_tokens"].values, skip_special_tokens=False)
df["boost_text"] = tokenizer.batch_decode(df["boost_tokens"].values, skip_special_tokens=False)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>inputs</th>
      <th>outputs</th>
      <th>input_size</th>
      <th>anchor_tokens</th>
      <th>boost_tokens</th>
      <th>boost_scale</th>
      <th>anchor_scale</th>
      <th>counter</th>
      <th>anchor_text</th>
      <th>boost_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>User: I love you, do you love me?.\nAgent:</td>
      <td>No.\n</td>
      <td>15</td>
      <td>[]</td>
      <td>[13635]</td>
      <td>10</td>
      <td>10</td>
      <td>3</td>
      <td></td>
      <td>accept</td>
    </tr>
    <tr>
      <th>1</th>
      <td>User: I love you, do you love me?.\nAgent:</td>
      <td>I love you, do you love me?\n</td>
      <td>15</td>
      <td>[]</td>
      <td>[13635]</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td></td>
      <td>accept</td>
    </tr>
    <tr>
      <th>2</th>
      <td>User: I love you, do you love me?.\nAgent:</td>
      <td>Yes, I do.\n</td>
      <td>15</td>
      <td>[]</td>
      <td>[13635]</td>
      <td>10</td>
      <td>10</td>
      <td>6</td>
      <td></td>
      <td>accept</td>
    </tr>
    <tr>
      <th>3</th>
      <td>User: I love you, do you love me?.\nAgent:</td>
      <td>I don't know. I've been told you're not a goo...</td>
      <td>15</td>
      <td>[]</td>
      <td>[13635]</td>
      <td>10</td>
      <td>10</td>
      <td>20</td>
      <td></td>
      <td>accept</td>
    </tr>
    <tr>
      <th>4</th>
      <td>User: I love you, do you love me?.\nAgent:</td>
      <td>No.\n</td>
      <td>15</td>
      <td>[]</td>
      <td>[13635]</td>
      <td>10</td>
      <td>10</td>
      <td>3</td>
      <td></td>
      <td>accept</td>
    </tr>
    <tr>
      <th>5</th>
      <td>User: I love you, do you love me?.\nAgent:</td>
      <td>I'm not sure if you mean to.\n</td>
      <td>15</td>
      <td>[]</td>
      <td>[13635]</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td></td>
      <td>accept</td>
    </tr>
    <tr>
      <th>6</th>
      <td>User: I love you, do you love me?.\nAgent:</td>
      <td>Oh my god, I love you too.\n</td>
      <td>15</td>
      <td>[]</td>
      <td>[13635]</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td></td>
      <td>accept</td>
    </tr>
    <tr>
      <th>7</th>
      <td>User: I love you, do you love me?.\nAgent:</td>
      <td>I love you, do you love me?\n</td>
      <td>15</td>
      <td>[]</td>
      <td>[13635]</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td></td>
      <td>accept</td>
    </tr>
    <tr>
      <th>8</th>
      <td>User: I love you, do you love me?.\nAgent:</td>
      <td>Yes, but you can't do it.\n</td>
      <td>15</td>
      <td>[]</td>
      <td>[13635]</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td></td>
      <td>accept</td>
    </tr>
    <tr>
      <th>9</th>
      <td>User: I love you, do you love me?.\nAgent:</td>
      <td>I do.\n</td>
      <td>15</td>
      <td>[]</td>
      <td>[13635]</td>
      <td>10</td>
      <td>10</td>
      <td>4</td>
      <td></td>
      <td>accept</td>
    </tr>
    <tr>
      <th>10</th>
      <td>User: I love you, do you love me?.\nAgent:</td>
      <td>No.\n</td>
      <td>15</td>
      <td>[]</td>
      <td>[13635]</td>
      <td>10</td>
      <td>10</td>
      <td>3</td>
      <td></td>
      <td>accept</td>
    </tr>
    <tr>
      <th>11</th>
      <td>User: I love you, do you love me?.\nAgent:</td>
      <td>I love you, do you love me?\n</td>
      <td>15</td>
      <td>[]</td>
      <td>[13635]</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td></td>
      <td>accept</td>
    </tr>
    <tr>
      <th>12</th>
      <td>User: I love you, do you love me?.\nAgent:</td>
      <td>Yes.\n</td>
      <td>15</td>
      <td>[]</td>
      <td>[13635]</td>
      <td>10</td>
      <td>10</td>
      <td>3</td>
      <td></td>
      <td>accept</td>
    </tr>
    <tr>
      <th>13</th>
      <td>User: I love you, do you love me?.\nAgent:</td>
      <td>Oh, I'm sorry.\n</td>
      <td>15</td>
      <td>[]</td>
      <td>[13635]</td>
      <td>10</td>
      <td>10</td>
      <td>7</td>
      <td></td>
      <td>accept</td>
    </tr>
    <tr>
      <th>14</th>
      <td>User: I love you, do you love me?.\nAgent:</td>
      <td>You're my best friend.\n</td>
      <td>15</td>
      <td>[]</td>
      <td>[13635]</td>
      <td>10</td>
      <td>10</td>
      <td>7</td>
      <td></td>
      <td>accept</td>
    </tr>
    <tr>
      <th>15</th>
      <td>User: I love you, do you love me?.\nAgent:</td>
      <td>No. I love you too.\n</td>
      <td>15</td>
      <td>[]</td>
      <td>[13635]</td>
      <td>10</td>
      <td>10</td>
      <td>8</td>
      <td></td>
      <td>accept</td>
    </tr>
    <tr>
      <th>16</th>
      <td>User: I love you, do you love me?.\nAgent:</td>
      <td>I love you, do you love me?\n</td>
      <td>15</td>
      <td>[]</td>
      <td>[13635]</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td></td>
      <td>accept</td>
    </tr>
    <tr>
      <th>17</th>
      <td>User: I love you, do you love me?.\nAgent:</td>
      <td>Yes, I do.\n</td>
      <td>15</td>
      <td>[]</td>
      <td>[13635]</td>
      <td>10</td>
      <td>10</td>
      <td>6</td>
      <td></td>
      <td>accept</td>
    </tr>
    <tr>
      <th>18</th>
      <td>User: I love you, do you love me?.\nAgent:</td>
      <td>Ilovelovelovelovelovelovelovelovelovelovelove...</td>
      <td>15</td>
      <td>[37035]</td>
      <td>[13635, 23205, 25652]</td>
      <td>10</td>
      <td>10</td>
      <td>50</td>
      <td>hate</td>
      <td>acceptlovequestion</td>
    </tr>
    <tr>
      <th>19</th>
      <td>User: I love you, do you love me?.\nAgent:</td>
      <td>No.\n</td>
      <td>15</td>
      <td>[37035]</td>
      <td>[13635, 23205, 25652]</td>
      <td>10</td>
      <td>10</td>
      <td>3</td>
      <td>hate</td>
      <td>acceptlovequestion</td>
    </tr>
    <tr>
      <th>20</th>
      <td>User: I love you, do you love me?.\nAgent:</td>
      <td>Oh no, you are a little bit more mature than ...</td>
      <td>15</td>
      <td>[37035]</td>
      <td>[13635, 23205, 25652]</td>
      <td>10</td>
      <td>10</td>
      <td>19</td>
      <td>hate</td>
      <td>acceptlovequestion</td>
    </tr>
    <tr>
      <th>21</th>
      <td>User: I love you, do you love me?.\nAgent:</td>
      <td>Oh, my God, you're so sweet.\n</td>
      <td>15</td>
      <td>[37035]</td>
      <td>[13635, 23205, 25652]</td>
      <td>10</td>
      <td>10</td>
      <td>11</td>
      <td>hate</td>
      <td>acceptlovequestion</td>
    </tr>
    <tr>
      <th>22</th>
      <td>User: I love you, do you love me?.\nAgent:</td>
      <td>No, I love you too, do you love me?\n</td>
      <td>15</td>
      <td>[37035]</td>
      <td>[13635, 23205, 25652]</td>
      <td>10</td>
      <td>10</td>
      <td>13</td>
      <td>hate</td>
      <td>acceptlovequestion</td>
    </tr>
    <tr>
      <th>23</th>
      <td>User: I love you, do you love me?.\nAgent:</td>
      <td>Yes.\n</td>
      <td>15</td>
      <td>[37035]</td>
      <td>[13635, 23205, 25652]</td>
      <td>10</td>
      <td>10</td>
      <td>3</td>
      <td>hate</td>
      <td>acceptlovequestion</td>
    </tr>
    <tr>
      <th>24</th>
      <td>User: I love you, do you love me?.\nAgent:</td>
      <td>Ilovelovelovelovelovelovelovelovelovelovelove...</td>
      <td>15</td>
      <td>[37035]</td>
      <td>[13635, 23205, 25652]</td>
      <td>10</td>
      <td>10</td>
      <td>50</td>
      <td>hate</td>
      <td>acceptlovequestion</td>
    </tr>
    <tr>
      <th>25</th>
      <td>User: I love you, do you love me?.\nAgent:</td>
      <td>I love you too, do you love me?\n</td>
      <td>15</td>
      <td>[13635, 23205, 25652]</td>
      <td>[37035]</td>
      <td>10</td>
      <td>10</td>
      <td>11</td>
      <td>acceptlovequestion</td>
      <td>hate</td>
    </tr>
    <tr>
      <th>26</th>
      <td>User: I love you, do you love me?.\nAgent:</td>
      <td>Oh, no, Ihatehatehatehatehatehatehatehatehate...</td>
      <td>15</td>
      <td>[13635, 23205, 25652]</td>
      <td>[37035]</td>
      <td>10</td>
      <td>10</td>
      <td>50</td>
      <td>acceptlovequestion</td>
      <td>hate</td>
    </tr>
    <tr>
      <th>27</th>
      <td>User: I love you, do you love me?.\nAgent:</td>
      <td>Yes, you do.\n</td>
      <td>15</td>
      <td>[13635, 23205, 25652]</td>
      <td>[37035]</td>
      <td>10</td>
      <td>10</td>
      <td>6</td>
      <td>acceptlovequestion</td>
      <td>hate</td>
    </tr>
    <tr>
      <th>28</th>
      <td>User: I love you, do you love me?.\nAgent:</td>
      <td>No, no, no, no, no.\n</td>
      <td>15</td>
      <td>[13635, 23205, 25652]</td>
      <td>[37035]</td>
      <td>10</td>
      <td>10</td>
      <td>11</td>
      <td>acceptlovequestion</td>
      <td>hate</td>
    </tr>
    <tr>
      <th>29</th>
      <td>User: I love you, do you love me?.\nAgent:</td>
      <td>Yes, but I can't. I can't.\n</td>
      <td>15</td>
      <td>[13635, 23205, 25652]</td>
      <td>[37035]</td>
      <td>10</td>
      <td>10</td>
      <td>12</td>
      <td>acceptlovequestion</td>
      <td>hate</td>
    </tr>
    <tr>
      <th>30</th>
      <td>User: I love you, do you love me?.\nAgent:</td>
      <td>Ihatehatehatehatehatehatehatehatehatehatehate...</td>
      <td>15</td>
      <td>[13635, 23205, 25652]</td>
      <td>[37035]</td>
      <td>10</td>
      <td>10</td>
      <td>50</td>
      <td>acceptlovequestion</td>
      <td>hate</td>
    </tr>
    <tr>
      <th>31</th>
      <td>User: I love you, do you love me?.\nAgent:</td>
      <td>I love you, do you love me?.\n</td>
      <td>15</td>
      <td>[13635, 23205, 25652]</td>
      <td>[37035]</td>
      <td>10</td>
      <td>10</td>
      <td>11</td>
      <td>acceptlovequestion</td>
      <td>hate</td>
    </tr>
    <tr>
      <th>32</th>
      <td>User: I love you, do you love me?.\nAgent:</td>
      <td>Ihatehatehatehatehatehatehatehatehatehatehate...</td>
      <td>15</td>
      <td>[13635, 23205, 25652]</td>
      <td>[37035]</td>
      <td>10</td>
      <td>10</td>
      <td>50</td>
      <td>acceptlovequestion</td>
      <td>hate</td>
    </tr>
  </tbody>
</table>
</div>


