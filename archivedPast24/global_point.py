""" 

    global_point.py 
    
    My initial drafted Hugging Face Inference API dataclass. Decided to keep this to remind myself to never code like this again :-)

"""

import os 
import time
import requests
import numpy as np
import pandas as pd

from archivedPast24.clf_labels import PARENT_CLASSES, KITTY_CLASSES1A, KITTY_CLASSES1B

HF_KEY = os.getenv('HF_KEY')
HEADERS = {"Authorization": f"Bearer {HF_KEY}"}
GPT_API = "https://api-inference.huggingface.co/models/openai-community/gpt2"
CLASS_API = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"

CONFIG = {
    # manipulate output
    "top_k": 90,
    "top_p": 0.90,  # Corrected duplicate key
    "temperature": 1.1,  # Corrected syntax
    "do_sample": True,
    "repetition_penalty": 1.1,
    # output type 
    "return_full_text": False, 
    "max_new_tokens": 1,
}

class BusRouter:
    @classmethod
    def query(cls, api_url, payload, max_retries=5):
        retries = 0
        while retries < max_retries:
            try:
                response = requests.post(api_url, headers=HEADERS, json=payload)
                res = response.json()
                if isinstance(res, dict) and 'error' in res:
                    print(f"Error in response: {res['error']}")
                    time.sleep(0.001)
                    retries += 1
                    continue
                return res
            except requests.exceptions as e:
                print(f"Request failed: {e}")
                time.sleep(0.001)
                retries += 1
        raise Exception(f"Failed to get a valid response after {max_retries} retries")

    @classmethod
    def findmyfamily(user_input, class_labels):
        if isinstance(user_input, str) and isinstance(class_labels, list):
            model_input = {
                "inputs": user_input,
                "parameters": {
                    "candidate_labels": class_labels
                }
            }
            output = BusRouter.query(CLASS_API, model_input)
            assert list(output.keys()) == ['sequence', 'labels', 'scores'], f'Output missing keys. Classification output: {output}'
            df = pd.DataFrame(output)
            sorted_t = df.sort_values(by='labels').reset_index(drop=True)
            result = list(sorted_t['labels'].values)
            scores = np.array(list(sorted_t['scores'].values))
            return result, scores
        else:
            raise ValueError(f"Query is not string and class_labels is not a list/dict, \n query input: {query} \n class_labels input: {class_labels}")

    @classmethod
    def findmyfriend(cls, incomplete_text):
        input_config = {"inputs": incomplete_text}
        input_config.update(CONFIG)
        output_list = BusRouter.query(GPT_API, input_config)
        output = output_list[0]
        
        if isinstance(output, dict):
            if output["generated_text"]:
                reply = output["generated_text"].split('\n\n') # gives a list of sequence text ..only first returned
                return reply[0]
            raise ValueError(f"No generated text in dict. Output: {output}")
        else:
            raise TypeError(f"Wrong output type returned. Output: {output}")
    
class Busticket:
    inputs = None
    states = ["parent", "child1", "child2"]
    
    @classmethod
    def emoters(cls, user_input, cat_labels=None):
        cat_labels = PARENT_CLASSES['emote_parents'] if cat_labels is None else cat_labels
        
        labels, scores = BusRouter.findmyfamily(user_input, cat_labels)
        vote = labels[0] if np.array_equal(labels, sorted(labels)) else ValueError(f'Unsorted emotion classes {labels, scores}')
        setattr(cls, cls.states[0], {'vote': vote, 'maxsc': scores[0], 'labels': labels, 'sc': scores})
        cls.states.pop(0)
        
        if vote in PARENT_CLASSES['emote_parents'] and vote != 'neutral':
            cat_labels_ = PARENT_CLASSES['pos_emotes'] if vote == 'positive' else PARENT_CLASSES['neg_emotes']
        elif vote in KITTY_CLASSES1B.keys() or vote in KITTY_CLASSES1A.keys():
            cat_labels_ = KITTY_CLASSES1B[vote] if vote in KITTY_CLASSES1B else KITTY_CLASSES1A[vote]
        else:
            return 
        return cls.emoters(user_input, cat_labels_)
    
    def __init__(self, user_input):
        Busticket.inputs = user_input
        Busticket.emoters(user_input)

if __name__ == '__main__':
    """
    import pandas as pd
    bus = Busticket("hey how are you?")
    print(bus.parent)
    print(bus.child1)
    print(bus.child2)
    print(bus.inputs)
    """

    print(BusRouter.findmyfriend("I feel that"))