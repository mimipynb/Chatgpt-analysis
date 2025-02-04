# DialGPT - Conversational Chatbot on GPT2

This notebook explores different decoding strategies.

TODO:

- [x] Greedy Search
- [x] Beam Search
- [x] Contrastive Search (in other cloud)
- [x] Manipulating Logits Distributions (in other notebooks)

A useful exercise to keep in mind:
	•	Text generation relative to the user’s interest. For example, if the user is engaged in the conversation, the chat generates longer responses (leading to more frequent messages). Otherwise, if the user is less interested, the responses are shorter.”

Git Rep:
https://github.com/microsoft/DialoGPT?tab=readme-ov-file

HF Model Card:
https://huggingface.co/microsoft/DialoGPT-small


TODO:
- [ ] Preform more statistical analysis on the correlation between decoders and expression states of generative models. 

Good sources
- [Generative Aspect-Based Sentiment Analysis with Contrastive Learning and Expressive Structure](https://arxiv.org/pdf/2211.07743)
- [Generative Sentiment Analysis via Latent Category Distribution and Constrained Decoding](https://arxiv.org/pdf/2407.21560)
- [Bert Basics](https://arxiv.org/pdf/1911.00536)

**Note**

The following section includes exploratory analysis on `classes` and `scripts` to configure key components in controlling streaming chatbots during live sessions. The experiment was setup to fetch user based feature classes for e.g. their mood per text sequence (positive, negative or neutral). Further work has been done in analytical notebooks under Naomi's git where TFIF (Term frequency) was used upon the categories rather than embedding the dialogue text. This is useful in the sense of approximating a kernel feature space relative to the certain type of individual - a valuable feature representation. This is useful in the sense that you are able to interpret the policy function in an agent's behaviour (which is predominant in their speech expression) as a latent space or hyperplane. In other words, a linear regression but you are viewing it form another angle - on 2D grid. 

Brief summary of setup 
- **Chat Session**: Stores/appends speech pairs in a chat session.
- **Generator**: GPT pipeline to return responses.
- **Online Chain**: A chain of classifiers for feature extraction in dialogue.
- **Experiment Sampler**: For experimenting with parameters like BeamSearch.

#### More self-notes but related to building my current agent, Naomi

Initially, the plan was to use two LLMs: one for roleplaying (e.g., Instruct LLaMA) and another for task-specific chats (e.g., DialGPT). The second model, DialGPT, is adjusted by decoding parameters inferred from chat sessions. Over time, these parameters are fine-tuned as more data is collected, with a threshold to maximize synthetic dataset generation. This fine-tuning may require significant compute resources, which may be postponed if needed.

For Naomi, this approach can be delayed as more features need to be collected outside Naomi's current use case.

### Reasoning:
Engagement levels are used to model human-like behavior in chat dialogues, such as predicting "double-texting" or when the bot sends multiple responses. This helps predict when the bot won't stop generating responses.

### Control Variables (Unit of Time):
1. **`time_window`**: Counts the number of speech pairs (user, agent) in a chat session.
2. **`session_window`**: Fine-tunes the neural network (e.g., LLaMA or ChatGPT).


```python
from transformers import pipeline, set_seed, GenerationConfig

set_seed(42)
# history = []
# user_role = 'user'
# agent_role = 'assistant'
model_card = "microsoft/DialoGPT-small"
generator = pipeline('text-generation', model=model_card, device='mps')
generator.tokenizer.pad_token_id = generator.tokenizer.eos_token_id 
generator.padding_side = 'left'
    
```


```python
generator.tokenizer.chat_template
```




    '{% for message in messages %}{{ message.content }}{{ eos_token }}{% endfor %}'




```python
generator.tokenizer.SPECIAL_TOKENS_ATTRIBUTES
```




    ['bos_token',
     'eos_token',
     'unk_token',
     'sep_token',
     'pad_token',
     'cls_token',
     'mask_token',
     'additional_special_tokens']




```python
generator(generator.tokenizer.apply_chat_template(
	[{'role': 'user', 'content': 'hey how are you?'}], tokenize=False, add_special_tokens=False
), return_full_text=False, max_new_tokens=30, do_sample=True)
```

    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.





    [{'generated_text': "My moms middle daughter's name"}]




```python
# %%writefile enforce_decode_params.py 

""" Generating synthetic dataset for testing Decoding Strategies. """

import numpy as np 
from typing import Literal 
from dataclasses import dataclass

SEED = 42
class Meter:
	""" Default Meter holding ranges / vals common and shared by all decoding strategies. """
 
	T = 30 # sampling param / spacer x_{i-1} - x_i
	min_k = 0.0 
	max_k = 100
	min_p = 0.0 
	max_p = 1.0 
	repitition_penalty_min = 1 
	repitition_penalty_max = 2
	temperature_min = 0.0 
	temperature_max = 2.0 
	alpha_min = 0.0 
	alpha_max = 1.0 
	beam_width_min = 2
	beam_width_max = 5 # inf
	# Default Attributes
	max_new_tokens=50
	num_return_sequences=1
	num_beams=1 # default 
	num_beam_groups=1 # default groups 

GEN_ARGS = GenerationConfig(
	do_sample=True,
	use_cache=False, 
	return_full_text=False, # False to only return the model's output text and not the users inputs 
	max_new_tokens=Meter.max_new_tokens,
)

class Sampler:
	GreedySearch = GenerationConfig(
		do_sample=False, 
		use_cache=False, 
		return_full_text=False,
		max_new_tokens=Meter.max_new_tokens,
		num_return_sequences=Meter.num_return_sequences, 
	)
	
	@dataclass 
	class BeamSearch:
		top_k = np.arange(Meter.min_k, stop=Meter.max_k, step=Meter.T) # [ 0, 15, 30, 45, 60, 75, 90]
		temperature = np.arange(Meter.temperature_min, Meter.temperature_max, step=Meter.T/100)
		num_beams = np.arange(Meter.beam_width_min, stop=Meter.beam_width_max, step=Meter.T) # or beam_widths = search area size per leaf step or n.o of branches to retain at each step
  
	@dataclass 
	class DiverseBeamSearch:
		num_beam_groups = 5

	@dataclass 
	class NucleusSearch:
		top_p = np.arange(Meter.min_p, stop=Meter.max_p, step=Meter.T) # [0.  , 0.15, 0.3 , 0.45, 0.6 , 0.75, 0.9 ]
		temperature = np.arange(Meter.temperature_min, Meter.temperature_max, step=Meter.T/100)

	@dataclass 
	class ContrastiveSearch(NucleusSearch):
		penalty_alpha = np.arange(Meter.alpha_min, Meter.alpha_max, Meter.T / 100)

	@staticmethod
	def exploreExploit(explore_type: Literal['Contrastive', 'BeamSearch', 'LogitScaler', 'NucleusSearch'] = 'BeamSearch'):
		""" Creates Exploiting and Exploring Feature Space.  """
  		
    	# Get the parameters from each class
		greedy = Sampler.GreedySearch.to_dict()
		explore = GEN_ARGS.to_dict().copy()
  
		if explore_type == 'BeamSearch':
			attr = {'top_k': Sampler.BeamSearch.top_k, 'temperature': Sampler.BeamSearch.temperature, 'num_beams': Sampler.BeamSearch.num_beams[0]}
		elif explore_type == 'NucleusSearch':
			attr = {'top_p': Sampler.NucleusSearch.top_p, 'temperature': Sampler.NucleusSearch.temperature}
		elif explore_type == 'Contrastive':
			attr = {'top_p': Sampler.ContrastiveSearch.top_p, 'temperature': Sampler.ContrastiveSearch.temperature, 'penalty_alpha': Sampler.ContrastiveSearch.penalty_alpha}
		else:
			raise ValueError('Retry pls. Unacceptable Decoding Strategy')
		
		explore.update(attr)
		return {'greedy': greedy, 'explore': explore}

def preprocess(pipe, user_inputs):
	return pipe.tokenizer.apply_chat_template(
		user_inputs, 
		tokenize=False, 
		add_special_tokens=False
	)
	
def chatbot(pipe, chat_inputs, **kwargs):
	""" Function contacting with the GPT Pipeline. """
	gen_args = GEN_ARGS.to_dict()
	if kwargs:
		gen_args.update(kwargs)
	
	response = pipe(chat_inputs, **gen_args)
 
	if not isinstance(response, list):
		raise ValueError(f"Expected list but got {response}")
	if not response[0].get('generated_text', None):
		raise ValueError(f"Expected 'generated_text' key in response but got {response}")
	
	print('Response:\n', response[0]['generated_text'])
	return response[0]['generated_text']

Sampler.exploreExploit()
```




    {'greedy': {'max_length': 20,
      'max_new_tokens': 50,
      'min_length': 0,
      'min_new_tokens': None,
      'early_stopping': False,
      'max_time': None,
      'stop_strings': None,
      'do_sample': False,
      'num_beams': 1,
      'num_beam_groups': 1,
      'penalty_alpha': None,
      'dola_layers': None,
      'use_cache': False,
      'cache_implementation': None,
      'cache_config': None,
      'return_legacy_cache': None,
      'temperature': 1.0,
      'top_k': 50,
      'top_p': 1.0,
      'min_p': None,
      'typical_p': 1.0,
      'epsilon_cutoff': 0.0,
      'eta_cutoff': 0.0,
      'diversity_penalty': 0.0,
      'repetition_penalty': 1.0,
      'encoder_repetition_penalty': 1.0,
      'length_penalty': 1.0,
      'no_repeat_ngram_size': 0,
      'bad_words_ids': None,
      'force_words_ids': None,
      'renormalize_logits': False,
      'constraints': None,
      'forced_bos_token_id': None,
      'forced_eos_token_id': None,
      'remove_invalid_values': False,
      'exponential_decay_length_penalty': None,
      'suppress_tokens': None,
      'begin_suppress_tokens': None,
      'forced_decoder_ids': None,
      'sequence_bias': None,
      'token_healing': False,
      'guidance_scale': None,
      'low_memory': None,
      'watermarking_config': None,
      'num_return_sequences': 1,
      'output_attentions': False,
      'output_hidden_states': False,
      'output_scores': False,
      'output_logits': None,
      'return_dict_in_generate': False,
      'pad_token_id': None,
      'bos_token_id': None,
      'eos_token_id': None,
      'encoder_no_repeat_ngram_size': 0,
      'decoder_start_token_id': None,
      'is_assistant': False,
      'num_assistant_tokens': 20,
      'num_assistant_tokens_schedule': 'constant',
      'assistant_confidence_threshold': 0.4,
      'prompt_lookup_num_tokens': None,
      'max_matching_ngram_size': None,
      'generation_kwargs': {},
      '_from_model_config': False,
      'transformers_version': '4.46.3',
      'return_full_text': False},
     'explore': {'max_length': 20,
      'max_new_tokens': 50,
      'min_length': 0,
      'min_new_tokens': None,
      'early_stopping': False,
      'max_time': None,
      'stop_strings': None,
      'do_sample': True,
      'num_beams': np.int64(2),
      'num_beam_groups': 1,
      'penalty_alpha': None,
      'dola_layers': None,
      'use_cache': False,
      'cache_implementation': None,
      'cache_config': None,
      'return_legacy_cache': None,
      'temperature': array([0. , 0.3, 0.6, 0.9, 1.2, 1.5, 1.8]),
      'top_k': array([ 0., 30., 60., 90.]),
      'top_p': 1.0,
      'min_p': None,
      'typical_p': 1.0,
      'epsilon_cutoff': 0.0,
      'eta_cutoff': 0.0,
      'diversity_penalty': 0.0,
      'repetition_penalty': 1.0,
      'encoder_repetition_penalty': 1.0,
      'length_penalty': 1.0,
      'no_repeat_ngram_size': 0,
      'bad_words_ids': None,
      'force_words_ids': None,
      'renormalize_logits': False,
      'constraints': None,
      'forced_bos_token_id': None,
      'forced_eos_token_id': None,
      'remove_invalid_values': False,
      'exponential_decay_length_penalty': None,
      'suppress_tokens': None,
      'begin_suppress_tokens': None,
      'forced_decoder_ids': None,
      'sequence_bias': None,
      'token_healing': False,
      'guidance_scale': None,
      'low_memory': None,
      'watermarking_config': None,
      'num_return_sequences': 1,
      'output_attentions': False,
      'output_hidden_states': False,
      'output_scores': False,
      'output_logits': None,
      'return_dict_in_generate': False,
      'pad_token_id': None,
      'bos_token_id': None,
      'eos_token_id': None,
      'encoder_no_repeat_ngram_size': 0,
      'decoder_start_token_id': None,
      'is_assistant': False,
      'num_assistant_tokens': 20,
      'num_assistant_tokens_schedule': 'constant',
      'assistant_confidence_threshold': 0.4,
      'prompt_lookup_num_tokens': None,
      'max_matching_ngram_size': None,
      'generation_kwargs': {},
      '_from_model_config': False,
      'transformers_version': '4.46.3',
      'return_full_text': False}}




```python
chat_input = generator.tokenizer.apply_chat_template(
    [
        {'role': 'system', 'content': 'Your name is Lucifer, chatting with a stranger online.'},
		{'role': 'user', 'content': 'hey what is your name? :)'},
		{'role': 'assistant', 'content': 'My name is Lucifer.'}, 
		{'role': 'user', 'content': 'How are you doing today?'}
	], 
    tokenize=False, 
    add_special_tokens=False
)

response = generator(chat_input, **GEN_ARGS.to_dict())

if not isinstance(response, list):
    raise ValueError(f"Expected list but got {response}")
if not response[0].get('generated_text', None):
    raise ValueError(f"Expected 'generated_text' key in response but got {response}")

print('Response:\n', response[0]['generated_text'])
```

    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
    Both `max_new_tokens` (=50) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)


    Response:
     Oh my gosh it's you!!! It even is, just a reminder this was your birthday yesterday. Happy Easter lt 3 I'm a simple man.



```python
#%%writefile agent_dial_utils.py

""" 

	agent_dial_utils.py 
	
	Script contains:
	- Utils for creating experiment's decoder params combinations and dataclasses 
	- Data handlers and Properties for Different Decoding strategies
 
"""

from enum import Enum 
from itertools import product 
from dataclasses import dataclass, field 

class SearchConfig(Enum):
	BeamSearch = (
		('top_k', 20, 100), 
		('temperature', 0.1, 2.0), 
		('num_beams', 1, 3)
	)

# aligns with the normal params for medium - neutral state
class DefaultParams(Enum):
	top_k = 50
	temperature = 1.0 
	num_beams = 1
	
@dataclass 
class BeamSearch:
	top_k: float = field(default=DefaultParams.top_k.value)
	temperature: float = field(default=DefaultParams.temperature.value)
	num_beams: int = field(default=1)
	
	@classmethod
	def packet(cls, top_k, temperature):
		return product(top_k, temperature, repeat=1)

def sampleParams(object, delta_t=3):
	decode_method = object.__class__.__qualname__ if object.__class__.__qualname__ != 'type' else object.__qualname__
	print('Objects type name:', decode_method) 
	# Find the corresponding configuration in SearchConfig Enum
	config = getattr(SearchConfig, decode_method, None)
	if config is None:
		raise ValueError(f"Unknown search method: {decode_method}")
	
	# Loop through the tuple (parameter name, min value, max value) and sample using np.linspace
	samples = {}
	for param, min_val, max_val in config.value:
		samples[param] = np.linspace(min_val, max_val, delta_t, dtype=type(min_val))

	return samples

beam_search = BeamSearch(0.9, 0.5, 1)
params = sampleParams(beam_search)
param_comb = list(BeamSearch.packet(params['top_k'], params['temperature']))

user_input = [{'role': 'user', 'content': 'Hey how are you today! :)'}]
chat_input = preprocess(generator, user_input)

result = []
for top_k_, temp_ in param_comb: 
	args = dict(top_k=top_k_.item(), temperature=temp_.item(), max_length=None)
	output = chatbot(pipe=generator, chat_inputs=user_input, **args)
	result.append(
		{
			'output': output, 
			'params': args, 
			'method': 'beam_search'
		}
	)

```

    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.


    Objects type name: BeamSearch


    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.


    Response:
     Hi and welcome!


    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.


    Response:
     Hello! How are you today?
    Response:
     Hey you


    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.


    Response:
     Hi and welcome!


    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.


    Response:
     I'm always there in spirit! Welcome to the sub!


    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.


    Response:
     OOC : Welcome c : What's been keeping u ceegojk from writing her usual speech after he was done hmu with his name so it could be a nice place...


    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.


    Response:
     Hi and welcome!


    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.


    Response:
     Oo I think I did it
    Response:
     What has your friends ever done these days because not caring so very much is generally enough for me haha! lt 3



```python
data = pd.DataFrame(result)
data
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
      <th>output</th>
      <th>params</th>
      <th>method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Hi and welcome!</td>
      <td>{'top_k': 20, 'temperature': 0.1, 'max_length'...</td>
      <td>beam_search</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Hello! How are you today?</td>
      <td>{'top_k': 20, 'temperature': 1.05, 'max_length...</td>
      <td>beam_search</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Hey you</td>
      <td>{'top_k': 20, 'temperature': 2.0, 'max_length'...</td>
      <td>beam_search</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hi and welcome!</td>
      <td>{'top_k': 60, 'temperature': 0.1, 'max_length'...</td>
      <td>beam_search</td>
    </tr>
    <tr>
      <th>4</th>
      <td>I'm always there in spirit! Welcome to the sub!</td>
      <td>{'top_k': 60, 'temperature': 1.05, 'max_length...</td>
      <td>beam_search</td>
    </tr>
    <tr>
      <th>5</th>
      <td>OOC : Welcome c : What's been keeping u ceegoj...</td>
      <td>{'top_k': 60, 'temperature': 2.0, 'max_length'...</td>
      <td>beam_search</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Hi and welcome!</td>
      <td>{'top_k': 100, 'temperature': 0.1, 'max_length...</td>
      <td>beam_search</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Oo I think I did it</td>
      <td>{'top_k': 100, 'temperature': 1.05, 'max_lengt...</td>
      <td>beam_search</td>
    </tr>
    <tr>
      <th>8</th>
      <td>What has your friends ever done these days bec...</td>
      <td>{'top_k': 100, 'temperature': 2.0, 'max_length...</td>
      <td>beam_search</td>
    </tr>
  </tbody>
</table>
</div>




```python
""" 

	Creates toy dataset to play with

"""
from datasets import load_dataset
from dataclasses import dataclass

tag_map = {
	"action": ["unknown", "inform", "question", "directive", "commissive"],
	"emotion": ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
}
emotion_map = {
	'happiness': 'positive', 'surprise': 'positive',
	'anger': 'negative', 'disgust': 'negative',
	'fear': 'negative', 'sadness': 'negative'
}

@dataclass
class Groupon:
	candidate_labels: list 
	
	@property 
	def N(self):
		return len(self.candidate_labels)

	@property
	def id2label(self):
		return dict(zip(range(self.N), self.candidate_labels))

	@property
	def label2id(self):
		return dict(zip(self.candidate_labels, range(self.N)))

	def label_input(self, item):
		if isinstance(item, str): 
			return self.label2id[item]
		elif isinstance(item, int): 
			return self.id2label[item]

def process_data(row):
	"""
	Process the row by splitting the dialog into 'user' and 'assistant' messages 
	and assigning corresponding emotions to 'u_big_emote' and 'a_big_emote'.
	"""
	# Initialize columns
	row['user'] = []
	row['assistant'] = []
	row['u_act'] = []
	row['a_act'] = []
	row['u_big_emote'] = []
	row['a_big_emote'] = []
	

	# Process the dialog and emotions
	for idx in range(0, len(row['dialog']) - 1, 2):
		row['user'].append(row['dialog'][idx])
		row['assistant'].append(row['dialog'][idx + 1])


		row['u_big_emote'].append(emotion_map.get(emoticon.label_input(row['emotion'][idx]), 'neutral'))
		row['a_big_emote'].append(emotion_map.get(emoticon.label_input(row['emotion'][idx + 1]), 'neutral'))
		row['u_act'].append(action.label_input(row['act'][idx]))
		row['a_act'].append(action.label_input(row['act'][idx+1]))
  
	return row 
```


```python
#action = Groupon(tag_map['action'])
#emoticon = Groupon(tag_map['emotion'])

#ds = load_dataset('li2017dailydialog/daily_dialog', split="train[:100]", trust_remote_code=True)
#ds = ds.map(process_data, remove_columns=['dialog', 'act', 'emotion'])
# ds.to_csv('sample_beam_search.csv', index=False)
data
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
      <th>user</th>
      <th>assistant</th>
      <th>u_act</th>
      <th>a_act</th>
      <th>u_big_emote</th>
      <th>a_big_emote</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[Say , Jim , how about going for a few beers a...</td>
      <td>[ You know that is tempting but is really not ...</td>
      <td>[directive, question, question, commissive, di...</td>
      <td>[commissive, question, directive, inform, comm...</td>
      <td>[neutral, neutral, neutral, positive, positive]</td>
      <td>[neutral, neutral, neutral, positive, positive]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[Can you do push-ups ? ,  Really ? I think tha...</td>
      <td>[ Of course I can . It's a piece of cake ! Bel...</td>
      <td>[question, question, inform]</td>
      <td>[inform, question, inform]</td>
      <td>[neutral, positive, neutral]</td>
      <td>[neutral, neutral, neutral]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[Can you study with the radio on ? ,  What is ...</td>
      <td>[ No , I listen to background music . ,  The r...</td>
      <td>[question, question]</td>
      <td>[inform, inform]</td>
      <td>[neutral, neutral]</td>
      <td>[neutral, neutral]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[Are you all right ? ,  Don't worry.He is an a...</td>
      <td>[ I will be all right soon . I was terrified w...</td>
      <td>[question, inform]</td>
      <td>[inform, inform]</td>
      <td>[neutral, neutral]</td>
      <td>[neutral, neutral]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[Hey John , nice skates . Are they new ? ,  Wh...</td>
      <td>[ Yeah , I just got them . I started playing i...</td>
      <td>[question, question, inform, inform]</td>
      <td>[inform, inform, question, directive]</td>
      <td>[neutral, neutral, neutral, neutral]</td>
      <td>[neutral, neutral, positive, positive]</td>
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
      <th>95</th>
      <td>[How was your education going on in Australia ...</td>
      <td>[ I'm going to graduate this summer . ,  I'm p...</td>
      <td>[question, question, directive]</td>
      <td>[inform, inform, commissive]</td>
      <td>[neutral, neutral, neutral]</td>
      <td>[neutral, neutral, neutral]</td>
    </tr>
    <tr>
      <th>96</th>
      <td>[Do you have any particular hobbies , Tom ? , ...</td>
      <td>[ Oh , yes . I love playing badminton , table ...</td>
      <td>[question, question, question, inform]</td>
      <td>[inform, inform, inform, inform]</td>
      <td>[neutral, neutral, neutral, neutral]</td>
      <td>[neutral, neutral, neutral, neutral]</td>
    </tr>
    <tr>
      <th>97</th>
      <td>[What ’ s the plot of your new movie ? ,  Did ...</td>
      <td>[ It ’ s a story about a policemen who is inve...</td>
      <td>[question, question, question, question, inform]</td>
      <td>[inform, inform, inform, inform, question]</td>
      <td>[neutral, neutral, neutral, neutral, positive]</td>
      <td>[neutral, neutral, neutral, neutral, positive]</td>
    </tr>
    <tr>
      <th>98</th>
      <td>[Who's that old lady trimming the trees ? ,  S...</td>
      <td>[ She's my grandma . ,  92 . ]</td>
      <td>[question, question]</td>
      <td>[inform, inform]</td>
      <td>[neutral, neutral]</td>
      <td>[neutral, neutral]</td>
    </tr>
    <tr>
      <th>99</th>
      <td>[Mom . My legs are killing me . ]</td>
      <td>[ Hold on . We will be successful right away . ]</td>
      <td>[directive]</td>
      <td>[commissive]</td>
      <td>[neutral]</td>
      <td>[neutral]</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 6 columns</p>
</div>




```python


""" 
	gpt_decode_params.py
	---------------------
	Experiment Runners / Helpers for defining optimal decoding Parameters for the GPT-2 Model relative to the following features:
 	- Current human speaker's features in observation window (e.g. chat window):
  		- Emotions
		- Engagement / Interests 
	- Agent's expected features (built from main i.e. myself):
		- Emotions 
		- Engagement / Interests 
	- Current Window combining Agent's and speaker's chat dynamics: 
		- Synchronicity of engagement 
 
	Objectives:
	- Define the optimal decoding parameters for the GPT-2 model based on the current speaker's features in the observation window.
	- Define the optimal decoding parameters for the GPT-2 model based on the agent's expected features.
	- Define the optimal decoding parameters for the GPT-2 model based on the current window combining the agent's and speaker's chat dynamics.
	- Preprocessors / Scalers for Human Features. 
"""

import pandas as pd 

time_window = 5 # 5 pairs of utterances 
FEATURES = {
	'emotion': ['positive', 'negative', 'neutral'], 
	'action': action, 
}

# caching handlers 
session = []

def build_history_space(session):
	""" Construct Priori's or initial state params for each feature space.  """
	pass 

def session_listener(func):
	def wrapper(*args, **kwargs):
		if len(func.featMetrics) > time_window:
			session.append(func.featMetrics)
			build_history_space(session)
			func.featMetrics = [] 
		return func(*args, **kwargs)
	print(f'Session Caches: {len(session)}')
	return wrapper

class ChatSession:
	""" 
 	Maps responses to the model outputs. 
 	Objective: A function that maps the engagement of the user. 
   	"""
	
	def __init__(self):
		self.history = []

	@property 
	def agent_history(self):
		diary = pd.DataFrame(self.history)
		history = diary[diary['role'] == 'assistant']['content'].values
		return history 

	@property 
	def user_history(self): 
		diary = pd.DataFrame(self.history)
		history = diary[diary['role'] == 'user']['content'].values 
		return history 

	def update_history(self, user_input, agent_input):
		""" Updates Pair of Speech Utterances. """
		self.history.append({
			'role': 'user', 
			'content': user_input 
		})
		self.history.append({
			'role': 'assistant', 
			'content': agent_input
		})
  
def run_episodes(num_episodes=2):
	# 2 chat dialogues ==> 2 strangers 
	chat = ChatSession()
	y_true = []
 
	for row_idx, row in ds.to_pandas().sample(num_episodes).iterrows():
		print(f'Adding data batch {row_idx}')
		for user_input, agent_input in zip(row['user'], row['assistant']):
			chat.update_history(user_input, agent_input)

		y_true.append({
			'u_act': row['u_act'], 
			'a_act': row['a_act'], 
			'u_big_emote': row['u_big_emote'], 
			'a_big_emote': row['a_big_emote']
		})

	return chat, y_true

# creates mock data 
chat, y_true = run_episodes(FEATURES)
```

    Adding data batch 79
    Adding data batch 37



```python
# creates inferencing chain 

class OnlineChain:
	def __init__(self, features, model_card: str = "facebook/bart-large-mnli", task_label: str = "zero-shot-classification"):
		self.clf = pipeline(task_label, model_card)
		self.features = {}
		for task_label, feat_label in features.items():
			feat = Groupon(feat_label) if not isinstance(feat_label, Groupon) else feat_label 
			self.features[task_label] = (self.classifier, feat)

	def classifier(self, inputs, candidate_labels):
		output = self.clf(inputs, candidate_labels)
		assert set(output.keys()) == set(['sequence', 'labels', 'scores']), f'Incorrect output from classifier: {output}'
		return dict(zip(output['labels'], output['scores']))

	def model_chain(self, inputs: str):
		result = {}
		for task, (model, groupon) in self.features.items():
			result[task] = model(inputs=inputs, candidate_labels=groupon.candidate_labels)
		return result 

chain = OnlineChain(features={'emotion': ['positive', 'negative', 'neutral']})
```

    Hardware accelerator e.g. GPU is available in the environment, but no `device` argument is passed to the `Pipeline` object. Model will be on CPU.



```python
example_result = data['output'].values
example_result 
```




    array(['Hi and welcome!', 'Hello! How are you today?', 'Hey you',
           'Hi and welcome!',
           "I'm always there in spirit! Welcome to the sub!",
           "OOC : Welcome c : What's been keeping u ceegojk from writing her usual speech after he was done hmu with his name so it could be a nice place...",
           'Hi and welcome!', 'Oo I think I did it',
           'What has your friends ever done these days because not caring so very much is generally enough for me haha! lt 3'],
          dtype=object)




```python
chain_output = {}
for item in example_result: 
    chain_output[item] = chain.model_chain(item)
```


```python
data['inferenced_emotion'] = data['output'].map(chain_output)

data['emote_label'] = data['inferenced_emotion'].apply(
    lambda x: max(x['emotion'], key=x['emotion'].get)
)
data['emote_score'] = data['inferenced_emotion'].apply(
    lambda x: max(x['emotion'].values())
)
```


```python
data['top_k'] = [i['top_k'] for i in data['params']]
data['temp'] = [i['temperature'] for i in data['params']]
data
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
      <th>output</th>
      <th>params</th>
      <th>method</th>
      <th>inferenced_emotion</th>
      <th>emote_label</th>
      <th>emote_score</th>
      <th>top_k</th>
      <th>temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Hi and welcome!</td>
      <td>{'top_k': 20, 'temperature': 0.1, 'max_length'...</td>
      <td>beam_search</td>
      <td>{'emotion': {'positive': 0.8815580010414124, '...</td>
      <td>positive</td>
      <td>0.881558</td>
      <td>20</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Hello! How are you today?</td>
      <td>{'top_k': 20, 'temperature': 1.05, 'max_length...</td>
      <td>beam_search</td>
      <td>{'emotion': {'positive': 0.6990548968315125, '...</td>
      <td>positive</td>
      <td>0.699055</td>
      <td>20</td>
      <td>1.05</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Hey you</td>
      <td>{'top_k': 20, 'temperature': 2.0, 'max_length'...</td>
      <td>beam_search</td>
      <td>{'emotion': {'positive': 0.5986918807029724, '...</td>
      <td>positive</td>
      <td>0.598692</td>
      <td>20</td>
      <td>2.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hi and welcome!</td>
      <td>{'top_k': 60, 'temperature': 0.1, 'max_length'...</td>
      <td>beam_search</td>
      <td>{'emotion': {'positive': 0.8815580010414124, '...</td>
      <td>positive</td>
      <td>0.881558</td>
      <td>60</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>I'm always there in spirit! Welcome to the sub!</td>
      <td>{'top_k': 60, 'temperature': 1.05, 'max_length...</td>
      <td>beam_search</td>
      <td>{'emotion': {'positive': 0.7329865097999573, '...</td>
      <td>positive</td>
      <td>0.732987</td>
      <td>60</td>
      <td>1.05</td>
    </tr>
    <tr>
      <th>5</th>
      <td>OOC : Welcome c : What's been keeping u ceegoj...</td>
      <td>{'top_k': 60, 'temperature': 2.0, 'max_length'...</td>
      <td>beam_search</td>
      <td>{'emotion': {'positive': 0.46241477131843567, ...</td>
      <td>positive</td>
      <td>0.462415</td>
      <td>60</td>
      <td>2.00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Hi and welcome!</td>
      <td>{'top_k': 100, 'temperature': 0.1, 'max_length...</td>
      <td>beam_search</td>
      <td>{'emotion': {'positive': 0.8815580010414124, '...</td>
      <td>positive</td>
      <td>0.881558</td>
      <td>100</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Oo I think I did it</td>
      <td>{'top_k': 100, 'temperature': 1.05, 'max_lengt...</td>
      <td>beam_search</td>
      <td>{'emotion': {'positive': 0.8012709617614746, '...</td>
      <td>positive</td>
      <td>0.801271</td>
      <td>100</td>
      <td>1.05</td>
    </tr>
    <tr>
      <th>8</th>
      <td>What has your friends ever done these days bec...</td>
      <td>{'top_k': 100, 'temperature': 2.0, 'max_length...</td>
      <td>beam_search</td>
      <td>{'emotion': {'negative': 0.49085739254951477, ...</td>
      <td>negative</td>
      <td>0.490857</td>
      <td>100</td>
      <td>2.00</td>
    </tr>
  </tbody>
</table>
</div>



#### Random Drafts build

**Quick recap of problem** 

**Objective**
The objective is to approximate the function between decoding parameters and dialogue engagement.

The following script considers tackling this method with monte-carlo sampling methods. However, I think it is safer to do a bit more reading and perform statistical tests like t-tests etc. before going full in with this component.

Predicting features I would like to target:
To make this problem feasible, it is a good idea to constrain the predicting features to binary classes. 

- **Engagement Intensity** refers to the stable state representing the dynamics of speech exchange between the user and agent e.g. bored vs extremely interested.
- **Emotion intensity** refers to responses that are minimal or absent in showing emotions (neutral) 


```python
""" 
	%%writefile draft_sampling.py 
 
	- **Engagement** refers to the stable state representing the dynamics of speech exchange between the user and agent.
	- Note: The function won't be singular for the ObservationSpace, especially after maximizing dataset re-generation likelihood.
	- Formally, the objective is to identify a function that maps engagement levels to optimal decoding parameter combinations.
	- Desired properties of this function:
	- Bidirectional
	- Range: [-1, 1]
 
"""

#from sklearn.linear_model import BayesianRidge 

@dataclass
class Gauss:
	mu: float = 0.0 
	std: float = 1.0

class KalmanFrame(object):
	def __init__(self, state_space_dim, action_space_dim, time_window: int = 5):
		self.n = state_space_dim 
		self.k = action_space_dim 
		self.t = time_window # e.g. state space at 0 will be the initial state observed, here, time_window (t) = 0 would be state space observed in the first 5 seconds/logs (default to 5)
  
	def gain(self):
		pass 
	
	def prior(self):
		pass 
	
	def posterior(self):
		pass 


class CurveFittingRoom:
    """ Workflow for inferencing tasks that deals with spaced time frames or where one of the controlling variables is over time (t). E.g. BayesRidgeRegression, Poly + LSQ"""
    pass 

class ObservationSpace:
	""" Start this at the beginning of the chat window. """
	
	def __init__(self):
	 
		# user and agent engagement level `[positive, negative, neutral]`. (i) Encoding to map the classification labels would make this a classification problem. (ii) Not encode the texts but focus specifically at the dialogue by interpreting as a frequency or resulting metrics from time-series transformation methods e.g. poly fitting / trig transformation 
  
		self.user_engagement = Gauss()
		self.agent_engagement = Gauss()
  
		# dialogue's engagement engagement level considering both user and agent (stacks of 5 i.e. [[A_tx_t - b_k], [A_tx_t - b_k], ...]) where k < t (or k = t-1) mean(axis=0) == float
		self.observation = CurveFittingRoom() # func to approximate the dialogue's transitioning window from previous to current engagement `units`
	
	
	def update(self, **kwargs):
		pass 
class Bandit:
	""" 
		Approximates a function between the decoding params (representing GPT's emotional/engagement control for now)
		Consideration's to mind:
			- One output per decoding combination of parameters may not be accurate as unique texts are generated per run, Monte Carlo may need to be used here or Posterior / Prior estimations can be considered here. 

			
	"""
	
	def __init__(self, decode_method: str = 'BeamSearch'):
		self.decode_method = decode_method 
		self.params = getattr(SearchConfig, decode_method, None) 
		if not self.params: 
			raise ValueError(f'ERror retrieving decoding method config: {decode_method}' )
		
		self.top_k = 
		self.temperature = 

		
```


```python
# This is the main class in controlling generative's models texts 

class AgentDial:
    def __init__(self, base_llm: pipeline, inference_chain: OnlineChain, decoder_params: dict):
        self.llm = base_llm # gpt
        self.inference_chain = inference_chain # returns the user features ... not the same as the chatting / dialogue helper agents chian 
        self.decoder_config = decoder_params 

    @property 
    def description(self):
        """
        Returns a detailed description of the agent's decoder configuration, outlining the decoder strategy 
        and the associated parameter values to be explored during decoding. 

        The decoder configuration consists of keys representing the decoder strategies, each paired with a 
        list of parameters (numpy arrays of values). The parameters will be used in combination, using 
        itertools.product, to explore the possible configurations during the agent's decoding process. 

        Example format for `decoder_params`:
            {
                'beam_search': [{'num_beams': np.array([5, 10]), 'length_penalty': np.array([1.0, 1.5])}],
                'sampling': [{'temperature': np.array([0.7, 1.0]), 'top_k': np.array([50, 100])}],
                'top_p': [{'top_p': np.array([0.9, 1.0])}],
            }

        The description provides an overview of these strategies and the potential parameter combinations 
        that the agent will try during inference.
        """
        return f"Decoder Strategies: {', '.join(self.decoder_config.keys())}, with respective parameter combinations: {list(product(*[list(val.values())[0] for val in strategy])) for strategy in self.decoder_config.values()}"
    
```

    58
    
    user           [What do you do in your free time , Nancy ? , ...
    assistant      [ Well , I like playing the violin . ,  About ...
    u_act                       [question, question, inform, inform]
    a_act                         [inform, inform, question, inform]
    u_big_emote               [neutral, neutral, positive, positive]
    a_big_emote               [neutral, neutral, positive, positive]
    Name: 58, dtype: object



```python
chat.history
```




    [{'role': 'user',
      'content': "You didn't ring me last night . You said you would . "},
     {'role': 'assistant',
      'content': " I'm sorry to have made you disappointed . "},
     {'role': 'user',
      'content': " That's all right . But why were you so rude to me at lunch . "},
     {'role': 'assistant',
      'content': ' Was I ? Sorry , I didn ’ t mean to be . I do apologize . '},
     {'role': 'user',
      'content': ' And why are you yarning now ? Are you bored ? '},
     {'role': 'assistant', 'content': " Forgive me darling . I'm very tired . "},
     {'role': 'user',
      'content': "What's the matter with you then ? You look miserable . "},
     {'role': 'assistant', 'content': " It's us . "},
     {'role': 'user', 'content': ' What do you mean by us . '},
     {'role': 'assistant', 'content': " Well , you always say you're busy . "},
     {'role': 'user', 'content': " That's right . "},
     {'role': 'assistant',
      'content': ' And you often go back to live with your parents and leave our son in the room by himself . '},
     {'role': 'user',
      'content': ' I ... I ... I miss my parents , also they miss me . '},
     {'role': 'assistant',
      'content': " Oh I remember , I cut terrible calls , and you didn't say anything about it . "},
     {'role': 'user', 'content': ' You mean I am groaned a few words ? '},
     {'role': 'assistant',
      'content': " Totally not . Perhaps it's about our marriage . "}]


