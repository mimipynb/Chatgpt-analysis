Sources:

[Paper on CAUSAL Transformer](https://arxiv.org/pdf/2106.01345)

FAQ

[HUGGING FACE RL WITH DL](https://huggingface.co/learn/deep-rl-course/en/unitbonus3/curriculum-learning)

[Ref Notebook on Google](https://colab.research.google.com/drive/1ezT24sogpVyr2HJLOvXHzjv61JZJ1gMT?usp=sharing#scrollTo=0MJJZEylVO-x)


```python
from datasets import load_dataset

data = load_dataset('li2017dailydialog/daily_dialog')

df = data['train'].to_pandas().head(1000)
df['dial_id'] = df.index.values
dfs = df.explode(['dialog', 'act', 'emotion'], ignore_index=True)
dfs['response'] = dfs.groupby('dial_id')['dialog'].shift(-1)
dfs['response_emote'] = dfs.groupby('dial_id')['emotion'].shift(-1)
```


```python
dfs
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
      <th>dialog</th>
      <th>act</th>
      <th>emotion</th>
      <th>dial_id</th>
      <th>response</th>
      <th>response_emote</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Say , Jim , how about going for a few beers af...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>You know that is tempting but is really not g...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>You know that is tempting but is really not g...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>What do you mean ? It will help us to relax .</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>What do you mean ? It will help us to relax .</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>Do you really think so ? I don't . It will ju...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Do you really think so ? I don't . It will ju...</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>I guess you are right.But what shall we do ? ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>I guess you are right.But what shall we do ? ...</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>I suggest a walk over to the gym where we can...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7374</th>
      <td>How about this sunday ?</td>
      <td>3</td>
      <td>4</td>
      <td>999</td>
      <td>Ok , cool .</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7375</th>
      <td>Ok , cool .</td>
      <td>4</td>
      <td>4</td>
      <td>999</td>
      <td>Good . I'll give you a call tonight .</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7376</th>
      <td>Good . I'll give you a call tonight .</td>
      <td>1</td>
      <td>4</td>
      <td>999</td>
      <td>No problem .</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7377</th>
      <td>No problem .</td>
      <td>1</td>
      <td>4</td>
      <td>999</td>
      <td>Bye .</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7378</th>
      <td>Bye .</td>
      <td>1</td>
      <td>4</td>
      <td>999</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>7379 rows × 6 columns</p>
</div>



## GPT2 using AutoModel loading


```python
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

set_seed(42)
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model.to('mps')
```




    GPT2LMHeadModel(
      (transformer): GPT2Model(
        (wte): Embedding(50257, 768)
        (wpe): Embedding(1024, 768)
        (drop): Dropout(p=0.1, inplace=False)
        (h): ModuleList(
          (0-11): 12 x GPT2Block(
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (attn): GPT2SdpaAttention(
              (c_attn): Conv1D()
              (c_proj): Conv1D()
              (attn_dropout): Dropout(p=0.1, inplace=False)
              (resid_dropout): Dropout(p=0.1, inplace=False)
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): GPT2MLP(
              (c_fc): Conv1D()
              (c_proj): Conv1D()
              (act): NewGELUActivation()
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
        )
        (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (lm_head): Linear(in_features=768, out_features=50257, bias=False)
    )




```python
model.generation_config
```




    GenerationConfig {
      "bos_token_id": 50256,
      "eos_token_id": 50256
    }




```python
input_text = "User: hey how are you? \nAgent:"
inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs.input_ids.to('mps')
attn_ids = inputs.attention_mask.to('mps')
input_length = input_ids.size(1)

input_length, inputs
```




    (11,
     {'input_ids': tensor([[12982,    25, 17207,   703,   389,   345,    30,   220,   198, 36772,
                 25]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])})




```python
from transformers import BeamSearchScorer
```


```python
# Search beam params
num_beams = 3 # how many beams to trak during the viterbi algorithm
num_return_beams = 3 # what is returned after algorithm
```


```python
# instantiating a BeamSearchScorer
beam_scorer = BeamSearchScorer(
    batch_size = input_length,
    num_beams = num_beams,
    num_beam_hyps_to_keep = num_return_beams,
    length_penalty=1,
    do_early_stopping=True,
    device=model.device
)

```


```python
config = dict(
    max_new_tokens=50,
    num_return_sequences=1,
    top_k=50,
    top_p=0.75,
    num_beams=2,
    temperature=0.9,
    stop_strings=["Agent:", "User:", "\n"],
    do_sample=True,
    tokenizer=tokenizer,
    output_attentions=False,
    output_logits=True,
    return_dict_in_generate=True,
    use_cache=False,
    early_stopping=True,
)
```


```python
output = model.generate(
    input_ids,
    attention_mask=attn_ids,
    **config
)

output_tokens = output.sequences[:, input_length:].squeeze(0)
output_tokens = output_tokens.cpu()       # Move to CPU for decoding if necessary

print(tokenizer.decode(output_tokens, skip_special_tokens=True))
```

    Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.


     
    Agent: 
    Agent: 
    Agent: 
    Agent: 
    Agent: 
    Agent: 
    Agent: 
    Agent: 
    Agent: 
    Agent: 
    Agent: 
    Agent: 
    



```python
torch.stack(output.attentions[0]).mean(dim=0).shape
```
