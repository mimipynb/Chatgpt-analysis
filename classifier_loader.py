""" 

	classifier_loader.py 
 
	Contains temporary dataclass to access multi-label classifier BART on hugging face from my bed. 

"""


import os 
import requests 
from typing import List, Any, Dict
from dataclasses import dataclass, field 

HF_TOKEN = os.getenv('HF_KEY', None)
if not HF_TOKEN:
	raise ValueError("No API Key found. Please set the HF_KEY environment variable.")

BASE_CLF_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"

@dataclass
class Classifier:
	"""
	A zero-shot classifier using Hugging Face's API.
	
	Attributes:
		candidate_labels (list): The list of labels to classify the input against.
		url (str): The endpoint for the Hugging Face API.
		model (str): The pre-trained model used for classification.
		task (str): The task type (default is zero-shot classification).
		_headers (dict): Authorization headers for API requests.
	"""
	candidate_labels: List[str] = field(default_factory=list)
	url: str = BASE_CLF_URL
	model: str = 'facebook/bart-large-mnli'
	task: str = 'zero-shot-classification'
	_headers: Dict[str, str] = field(init=False)

	def __post_init__(self):
		"""
		Post-initialization to validate and set up headers.
		"""
		if not self.candidate_labels:
			raise ValueError("candidate_labels cannot be empty. Provide at least one label.")
		
		self._headers = {'Authorization': f'Bearer {HF_TOKEN}'}

	def query(self, inputs: str) -> Dict[str, Any]:
		"""
		Sends a request to the Hugging Face API for zero-shot classification.

		Args:
			inputs (str): The text input to classify.

		Returns:
			dict: The response from the Hugging Face API, parsed as JSON.

		Raises:
			ValueError: If the inputs are invalid or the response is unsuccessful.
		"""
		if not isinstance(inputs, str):
			raise TypeError(f"Input must be a non-empty string. {inputs}")
		
		model_input = {
			'inputs': inputs,
			'parameters': {
				'candidate_labels': self.candidate_labels
			}
		}

		try:
			response = requests.post(self.url, headers=self._headers, json=model_input)
			response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
			return response.json()
		except requests.exceptions.RequestException as e:
			raise RuntimeError(f"Request failed: {e}") from e
