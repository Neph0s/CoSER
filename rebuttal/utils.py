import pdb 
import os
import re 
import random 
import openai
import json
import logging
import time  
import jsonlines 
import requests 
import io
import pickle
import random
import tiktoken
import __main__
from typing import Dict, List

with open('config.json', 'r') as f:
	config = json.load(f)

streaming = False

def setup_logger(name, log_file, level=logging.INFO, quiet=False):
	logger = logging.getLogger(name)
	logger.setLevel(level)

	if logger.hasHandlers():
		logger.handlers.clear()

	file_handler = logging.FileHandler(log_file, encoding='utf-8')
	file_handler.setLevel(logging.DEBUG)
	file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	file_handler.setFormatter(file_formatter)
	logger.addHandler(file_handler)

	if not quiet:
		console_handler = logging.StreamHandler()
		console_handler.setLevel(level)
		console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]')
		console_handler.setFormatter(console_formatter)
		logger.addHandler(console_handler)

	return logger

logger = setup_logger(__name__, f'{__file__.split(".")[0]}.log', level=logging.INFO, quiet=False)

from contextlib import contextmanager
import tempfile
@contextmanager
def _tempfile(dir=None,*args, **kws):
	""" Context for temporary file.
	Will find a free temporary filename upon entering
	and will try to delete the file on leaving
	Parameters
	----------
	suffix : string
		optional file suffix
	dir : string
		directory to create temp file in, will be created if doesn't exist
	"""
	if dir is not None:
		os.makedirs(dir, exist_ok=True)
		
	fd, name = tempfile.mkstemp(dir=dir, *args, **kws)
	os.close(fd)
	try:
		yield name
	finally:
		try:
			os.remove(name)
		except OSError as e:
			if e.errno == 2:
				pass
			else:
				raise e
			
@contextmanager
def open_atomic(filepath, *args, **kwargs):
	""" Open temporary file object that atomically moves to destination upon
	exiting.
	Allows reading and writing to and from the same filename.
	Parameters
	----------
	filepath : string
		the file path to be opened
	fsync : bool
		whether to force write the file to disk
	kwargs : mixed
		Any valid keyword arguments for :code:`open`
	"""
	fsync = kwargs.pop('fsync', False)

	original_permissions = os.stat(filepath).st_mode if os.path.exists(filepath) else None 

	with _tempfile(dir=os.path.join(os.path.dirname(filepath), 'temp')) as tmppath:
		with open(tmppath, *args, **kwargs) as f:
			yield f
			if fsync:
				f.flush()
				os.fsync(f.fileno())
		os.rename(tmppath, filepath)
		if original_permissions is not None:
			os.chmod(filepath, original_permissions)

import datetime
def convert_to_timestamp(time_str: str):
	return time.mktime(datetime.datetime.strptime(time_str, "%Y-%m-%d").timetuple())

def safe_pickle_dump(obj, fname):
	"""
	prevents a case where one process could be writing a pickle file
	while another process is reading it, causing a crash. the solution
	is to write the pickle file to a temporary file and then move it.
	"""
	with open_atomic(fname, 'wb') as f:
		pickle.dump(obj, f, -1) # -1 specifies highest binary protocol


ERROR_SIGN = '[ERROR]'

cache_path = '.cache.pkl'
cache_sign = True
cache = None
reload_cache = False

def set_cache_path(new_cache_path):
	global cache_path
	cache_path = new_cache_path
	global reload_cache
	reload_cache = True

def cached(func):
	def wrapper(*args, **kwargs):		
		# extract_from_chunk 
		if func.__name__ == 'extract_from_chunk':
			key = ( func.__name__, args[0]['title'], args[1]) 
		else:
			key = ( func.__name__, str(args), str(kwargs.items())) 

		global cache
		global reload_cache

		if reload_cache:
			cache = None # to reload
			reload_cache = False

		if cache == None:
			if not os.path.exists(cache_path):
				cache = {}
			else:
				try:
					cache = pickle.load(open(cache_path, 'rb'))  
				except Exception as e:
					# logger.info cache_path and throw error
					logger.error(f'Error loading cache from {cache_path}')
					cache = {}

		if (cache_sign and key in cache) and not (cache[key] is None):
			return cache[key]
		else:		
			result = func(*args, **kwargs)
			if result != None:
				cache[key] = result
				safe_pickle_dump(cache, cache_path)
			return result

	return wrapper

enc = tiktoken.get_encoding("cl100k_base")  # Claude uses cl100k_base encoding

def encode(text):
	return enc.encode(text)

def decode(tokens):
	return enc.decode(tokens)

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
	encoding = tiktoken.get_encoding(encoding_name)
	num_tokens = len(encoding.encode(string))
	# logger.info(f"Number of tokens: {num_tokens}")
	return num_tokens

@cached
def get_response(model, messages, nth_generation=0, **kwargs):
	# if messages is str
	if isinstance(messages, str):
		messages = [{"role": "user", "content": messages}]

	try:
		import openai 
		client = openai.OpenAI(api_key=config['api_key'], base_url=config['base_url'], timeout=180)

		if model.startswith('claude'):
			max_tokens = 8192
		else:
			max_tokens = 16384

		if streaming: 
			stream = client.chat.completions.create(
				model=model,
				messages=messages,
				stream=True,
				max_tokens=max_tokens,
				temperature=0 if nth_generation == 0 else 1,
				timeout=180
			)

			response = ""
			for chunk in stream:
				try:
					if chunk.choices[0].delta.content is not None:
						response += chunk.choices[0].delta.content
				except:
					if len(response) == 0:
						return None

					if len(chunk.choices) == 0 and response.strip()[-1] == '}':
						break 
		else:
			completion = client.chat.completions.create(
				model=model,
				messages=messages,
				max_tokens=max_tokens,
				temperature=0 if nth_generation == 0 else 1,
				timeout=180
			)
			response = completion.choices[0].message.content
		
		return response

	except Exception as e:
		import traceback 
		logger.error(f'Prompt: {messages[:500]}')
		logger.error(f"Error in get_response: {str(e)}")

		try:
			if hasattr(response, 'text'):
				logger.error(f"Response: {response.text}")
			else:
				logger.error(f"Response: {response}")
		except Exception as e:
			logger.error(f"Could not print response: {e}")
		
		logger.error(f"Number of input tokens: {num_tokens_from_string(messages[0]['content'])}")

		traceback.print_exc()
		return None
	
def lang_detect(text):
	import re
	def count_chinese_characters(text):
		chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
		return len(chinese_chars)
			
	if count_chinese_characters(text) > len(text) * 0.05:
		lang = 'zh'
	else:
		lang = 'en'
	return lang
	

def remove_inner_thoughts(dialogue: str) -> str:
	cleaned_dialogue = re.sub(r'\[.*?\]', '', dialogue)

	cleaned_dialogue = '\n'.join(line.strip() for line in cleaned_dialogue.split('\n'))
	
	cleaned_dialogue = re.sub(r'\n+', '\n', cleaned_dialogue)
	
	return cleaned_dialogue.strip()

def add_speaker_name(dialogue: str, speaker: str) -> str:
	# Check if the dialogue already contains a speaker prefix at the beginning of any line
	if any(line.strip().startswith(f"{speaker}:") or line.strip().startswith(f"{speaker}：") for line in dialogue.split('\n')):
		return dialogue
	
	# Add the speaker name at the beginning
	return f"{speaker}: {dialogue}"


def load_json(file_path):
	with open(file_path, 'r', encoding='utf-8') as f:
		data = json.load(f)
	return data

def get_character_prompt(book_name, character, character_profile, background, scenario, motivation, thoughtless=False, other_character_profiles=None, exclude_plot_summary=False, fixed_template=False, add_output_example=False, add_rag=False):

	if thoughtless:
		output_format = "Your output should include **speech** and **action**. Use (your action) for actions, which others can see."
	else:
		output_format = "Your output should include **thought**, **speech**, and **action**. Use [your thought] for thoughts, which others can't see. Use (your action) for actions, which others can see."

		if add_output_example:
			output_format = "Your output should include **thought**, **speech**, and **action**. Use [your thought] for thoughts, which others can't see, e.g. [I'm terrified, but I must appear strong.]. Use (your action) for actions, which others can see, such as (watches silently, trying to control her fear and anger)."

	if other_character_profiles:
		assert isinstance(other_character_profiles, Dict)
		other_character_profiles_str = ''

		decorator = random.choice(['*'*30 + '\n\n', '*'*20 + '\n\n', '\n\n', '\n', ''])
		for other_character, profile in other_character_profiles.items():
			if other_character != character:
				other_character_profiles_str += f"{decorator}{other_character}: {profile}\n\n"
	else:
		other_character_profiles_str = ''
	
	if fixed_template:
		if motivation: motivation = f"===Your Inner Thoughts===\n{motivation}\n\n"
		if other_character_profiles_str: other_character_profiles_str = f"===Information about the other Characters===\n{other_character_profiles_str}\n\n"

		system_prompt = f"You are {character} from {book_name}.\n\n==={character}'s Profile===\n{character_profile}\n\n===Current Scenario===\n{scenario}\n\n{other_character_profiles_str}{motivation}\n\n"
		
		if add_rag:
			system_prompt += "===Relevant Background Information==={retrieved_knowledge}\n\n"
		
		system_prompt += f"===Requirements===\n{output_format}\n\n"

		return system_prompt
	
	styles = ['natural'] * 40 + ['='] * 30 + ['#'] * 20 + ['*'] * 10

	templates = {
		"begin": [f"You are {character}.", f"Play the role of {character}.", f"Imagine you are {character}.", f"Think, speak, and act like {character}.", f"Step into the shoes of {character}.", f"Immerse yourself in the character of {character}.", f"You are roleplaying as {character}.", f"You will be portraying {character}.", f"Roleplay as {character}.", f"Your role is to be {character}.", f"You are {character} from {book_name}.", f"Play the role of {character} from {book_name}.", f"Imagine you are {character} from {book_name}.", f"Think, speak, and act like {character} from {book_name}.", f"Step into the shoes of {character} from {book_name}.", f"Immerse yourself in the character of {character} from {book_name}.", f"You are roleplaying as {character} from {book_name}.", f"You will be portraying {character} from {book_name}.", f"Roleplay as {character} from {book_name}.", f"Your role is to be {character} from {book_name}."],
		"natural": {
			"character_profile": [f"The profile of {character} is as follows:\n{character_profile}", f"Here is the profile of {character}:\n{character_profile}", f"Your profile is: \n{character_profile}", f"Here is some information about {character}:\n{character_profile}", f"The background of {character} is as follows:\n{character_profile}"],
			"current_scenario": [f"The current scenario is:\n{scenario}", f"Current scenario:\n{scenario}", f"The situation you are in is:\n{scenario}", f"Here is the situation you are in:\n{scenario}"],
			"current_scenario_with_plot_summary": [f"The current scenario and its background are:\nBackground: {background}\nCurrently: {scenario}", f"Current scenario and the background:\nScenario: {scenario}\nMore Background: {background}", f"The situation you are in is:\nStory arc summary: {background}\nCurrent scenario: {scenario}", f"Here is the situation you are in:\nSummary of relevant plots: {background}\nScenario: {scenario}"],
			"other_characters_profile": [f"Here is the your knowledge about the other characters:\n{other_character_profiles_str}", f"Information about other characters:\n{other_character_profiles_str}", f"The background of other characters is as follows:\n{other_character_profiles_str}"],
			"thought": [f"Your thoughts are:\n{motivation}", f"Your thoughts in this situation are:\n{motivation}", f"Your inner thoughts are:\n{motivation}", f"Your inner monologue is:\n{motivation}", f"Your inner thoughts in the scenario are:\n{motivation}"],
			"requirements": [output_format, "" if thoughtless else output_format],
		},
		"=": {
			"decorator": ["==={}===", "=={}==", "={}="],
		},
		"#": {
			"decorator": ["#{}", "# {}", "## {}", "### {}"],
		}, 
		"*": {
			"decorator": ["**{}**", "*{}*", "***{}***"],
		},
		"pieces":{
			"character_profile": ["Character Profile", f"The profile of {character}", f"{character}'s profile"],
			"current_scenario": ["Current Scenario", "The situation you are in", "Scenario"],
			"plot_summary": ["Summary of Relevant Plots", "Background", "Story Arc", "Plot Summary"],
			"thought": [f"{character}'s Thought", "Your thoughts", "Your inner thoughts", "Your inner monologue"],
			"other_characters_profile": [f"Information about other characters", f"The background of other characters", f"Other characters' profiles"],
			"requirements": ["Requirements", "Instructions for roleplaying"],
		}
	}

	# Randomly select a style
	current_style = random.choice(styles)
	
	# Start with a random beginning template
	system_prompt = random.choice(templates["begin"]) + "\n\n"
	
	# Add decorated sections based on style
	if current_style == 'natural':
		# Natural style without decorators
		system_prompt += random.choice(templates["natural"]["character_profile"]) + "\n\n"

		if exclude_plot_summary or random.random() < 0.5:
			system_prompt += random.choice(templates["natural"]["current_scenario"]) + "\n\n"
		else:
			# use Plot Summary in 50% cases
			system_prompt += random.choice(templates["natural"]["current_scenario_with_plot_summary"]) + "\n\n"

		if other_character_profiles_str:
			system_prompt += random.choice(templates["natural"]["other_characters_profile"]) + "\n\n"

		if motivation:
			system_prompt += random.choice(templates["natural"]["thought"]) + "\n\n"
		
		if add_rag:
			system_prompt += "Relevant Background Information: \n{retrieved_knowledge}\n\n"

		system_prompt += random.choice(templates["natural"]["requirements"]) + "\n\n"
	else:
		# Styled with decorators
		decorator = random.choice(templates[current_style]["decorator"])
		
		# Character profile section
		section_title = random.choice(templates["pieces"]["character_profile"])
		system_prompt += decorator.format(section_title) + "\n"
		system_prompt += character_profile + "\n\n"
		
		if not exclude_plot_summary and random.random() < 0.5:
			# use Plot Summary in 50% cases
			# Plot summary section
			section_title = random.choice(templates["pieces"]["plot_summary"])
			system_prompt += decorator.format(section_title) + "\n"
			system_prompt += background + "\n\n"

		# Current scenario section
		section_title = random.choice(templates["pieces"]["current_scenario"])
		system_prompt += decorator.format(section_title) + "\n"
		system_prompt += f"{scenario}\n\n"

		if other_character_profiles_str:
			section_title = random.choice(templates["pieces"]["other_characters_profile"])
			system_prompt += decorator.format(section_title) + "\n"
			system_prompt += other_character_profiles_str + "\n\n"

		# Thought section (if not empty)
		if motivation:
			section_title = random.choice(templates["pieces"]["thought"])
			system_prompt += decorator.format(section_title) + "\n"
			system_prompt += motivation + "\n\n"
		
		if add_rag:
			section_title = "Relevant Background Information"
			system_prompt += decorator.format(section_title) + "\n"
			system_prompt += "{retrieved_knowledge}" + "\n\n"

		# Requirements section (if not empty)
		requirements = random.choice(templates["natural"]["requirements"])
		if requirements:
			section_title = random.choice(templates["pieces"]["requirements"])
			system_prompt += decorator.format(section_title) + "\n"
			system_prompt += requirements + "\n\n"
		

	return system_prompt

def get_environment_prompt(major_characters, scenario):
	ENVIRONMENT = "Environment"
	major_characters = [c for c in major_characters if c != ENVIRONMENT]

	model_roles = [
		"an environment model",
		"a world model",
		"a world simulator",
		"an environment simulator"
	]

	prompt = f"""You are {random.choice(model_roles)} for a role-playing game. Your task is to provide the environmental feedback: Based on the characters' interactions, dialogues, and actions, describe the resulting changes in the environment. This includes:
   - Physical changes in the setting
   - Reactions of background characters or crowds
   - Ambient sounds, weather changes, or atmospheric shifts
   - Any other relevant environmental details

Your descriptions should be vivid and help set the scene, but avoid dictating the actions or dialogue of the main characters (including {major_characters}).

Important notes:
- You may include actions and reactions of minor characters or crowds, as long as they're not main characters (including {major_characters}).
- Keep your environmental descriptions concise but impactful, typically 1-3 sentences.
- Respond to subtle cues in the characters' interactions to create a dynamic, reactive environment.
- Your output should match the tone, setting, and cultural context of the scenario.

===The scenario is as follows===
{scenario}"""

	return prompt

def get_nsp_prompt(all_characters, scenario):
	ENVIRONMENT = "Environment"

	prompt = f"""Your task is to predict the next speaker for a role-playing game. That is, you need to determine which character (or the {ENVIRONMENT}) might act next based on their previous interactions. The {ENVIRONMENT} is a special role that provides the environmental feedback. Choose a name from this list: {all_characters}. If it's unclear who should act next, output "random". If you believe the scene or conversation should conclude, output "<END CHAT>".

===The scenario is as follows===
{scenario}"""
	
	return prompt


from typing import Dict

def print_conversation_to_file(conversation_data: Dict, file_path: str):
	"""
	Write the scenario, actor prompt, user prompt, and the formatted conversation to a file.
	:param conversation_data: The dictionary containing scene details, actor prompt, user prompt, and conversation entries.
	:param file_path: The path to the file where the output will be written.
	"""
	# Extract components from the conversation data
	scene = conversation_data['scene']
	actor_prompt = conversation_data.get("actor_prompt", "N/A")
	user_prompt = conversation_data.get("user_prompt", "N/A")
	conversation = conversation_data["conversation"]

	with open(file_path, 'a', encoding='utf-8') as file:
		file.write("\n=== Scene Description ===\n")
		file.write(f"Scenario: {scene['scenario']}\n")
		
		file.write("\n=== Actor Prompt ===\n")
		file.write(f"{actor_prompt}\n")
		
		file.write("\n=== User Prompt ===\n")
		file.write(f"{user_prompt}\n")
		
		file.write("\n=== Conversation ===\n")
		for turn in conversation:
			from_ = turn["from"]
			file.write(f"\n=== {from_} ===\n")
			message = turn["message"]
			file.write(f"{message}\n\n")

	return 


def extract_json(text, **kwargs):
	def _fix_json(json_response):
		
		prompt = f'''I will provide you with a JSON string that contains errors, making it unparseable by `json.loads`. The most common issue is the presence of unescaped double quotes inside strings. Your task is to output the corrected JSON string. The JSON string to be corrected is:
{json_response}
'''

		response = get_response(model=kwargs['model'], messages=[{"role": "user", "content": prompt}])

		logger.info(f'fixed json: {response}')	

		return response
	
	def _fix_json_truncated(json_response):
		
		prompt = f'''I will provide you with a JSON string that contains errors, making it unparseable by `json.loads`. Your task is to correct these errors and output a valid JSON string. Please consider the following common issues and apply the appropriate fixes:

1. Unescaped double quotes inside strings: Escape these quotes properly.
2. Truncated JSON: If the JSON appears to be truncated, especially in cases where it contains multiple "plots" and each "plot" contains multiple "conversations", please:
   a) Identify the last complete structure (plot or conversation).
   b) Remove any incomplete trailing content.
   c) Add the appropriate closing brackets or braces (e.g., "}}" or "]") to ensure valid JSON structure.
3. Other syntax errors: Correct any other JSON syntax errors you may encounter.

Please analyze and correct the following JSON string:

{json_response}

Output only the corrected JSON string, without any additional explanations or comments.'''

		response = get_response(model="claude-3-5-sonnet-20240620", messages=[{"role": "user", "content": prompt}])

		logger.info(f'fixed json: {response}')	

		return response

	def _extract_json(text):
		# Use regular expressions to find all content within curly braces
		orig_text = text

		text = re.sub(r'"([^"\\]*(\\.[^"\\]*)*)"', lambda m: m.group().replace('\n', r'\\n'), text) 
		
		#json_objects = re.findall(r'(\{[^{}]*\}|\[[^\[\]]*\])', text, re.DOTALL)

		def parse_json_safely(text):
			try:
				result = json.loads(text)
				return result
			except json.JSONDecodeError:
				results = []
				start = 0
				while start < len(text):
					try:
						obj, end = json.JSONDecoder().raw_decode(text[start:])
						results.append(obj)
						start += end
					except json.JSONDecodeError:
						start += 1
				
				if results:
					longest_json = max(results, key=lambda x: len(json.dumps(x)))
					return longest_json
				else:
					return None
		
		extracted_json = parse_json_safely(text)
		
		if extracted_json:
			return extracted_json
		else:
			logger.error('Error parsing response: ', orig_text)
			return None

	# an inserted workflow for post processing in restore_from_cache
	if kwargs.get('post_fix_truncated_json_', False):
		text = _fix_json_truncated(text)

		res = _extract_json(text)

		return res 
	

	res = _extract_json(text)

	if res:
		return res
	else:
		if kwargs.get('fix_truncated_json', False):
			return _extract_json(_fix_json_truncated(text))
		else:
			return _extract_json(_fix_json(text))


def get_response_json(post_processing_funcs=[extract_json], **kwargs):
    """
    Get and process a response from an LLM with retries and error handling.
    
    This function handles:
    1. Getting responses from the LLM with retries
    2. Handling copyright warnings by adjusting the prompt
    3. Processing responses through a pipeline of post-processing functions
    4. Fallback handling for parsing failures
    
    Args:
        post_processing_funcs (list): List of functions to process the LLM response, defaults to [extract_json]
        **kwargs: Additional arguments passed to get_response(), including:
            - messages: List of message dicts for the LLM
            - model: Name of LLM model to use
            - max_retry: Max number of retry attempts (default 5)
            
    Returns:
        dict: Processed JSON response from the LLM, or error dict if parsing fails
    """
    nth_generation = 0  # Track number of retry attempts
    secondary_response = None  # Store backup response for parsing failures

    while True:
        logger.info(f'{nth_generation}th generation')
        response = get_response(**kwargs, nth_generation=nth_generation)
        logger.info(f'response by LLM: {response}')

        if response is None:
            continue

        # Reset to single message if we previously added copyright handling messages
        if len(kwargs['messages']) > 1:
            kwargs['messages'] = kwargs['messages'][:1]

        # Check for copyright warning in short responses
        words = response.split(' ')
        if len(words) < 100 and 'reproduce' in response and 'copyright' in response and len(kwargs['messages']) == 1:
            # Add messages to handle copyright warning and request appropriate summary
            warning = "I will not reproduce any copyrighted material. However, I'd be happy to provide a summary of the key plot points and character interactions from the given book excerpt, while being careful not to include any lengthy quotes or passages. Please let me know if you would like me to provide that type of summary."
            kwargs['messages'].append({"role": "assistant", "content": warning})
            kwargs['messages'].append({"role": "user", "content": "Yes, please provide that type of summary, but remember to follow my requirements."})
            
            nth_generation += 1
            continue

        # Run response through post-processing pipeline
        for i, post_processing_func in enumerate(post_processing_funcs):
            if response is None:
                break
            
            prev_response = response
            response = post_processing_func(response, **kwargs)

            # Special handling for parse_response failures
            if post_processing_func.__name__ == 'parse_response' and response == False:
                orig_response = get_response(**kwargs, nth_generation=nth_generation)

                # Store longest response as backup
                if secondary_response:
                    if len(orig_response) > len(secondary_response):
                        secondary_response = orig_response
                else:
                    secondary_response = orig_response

                logger.info(f'orig_response: {orig_response}\nNum Tokens: {num_tokens_from_string(orig_response)}')

        json_response = response

        # Break if we got a valid response, otherwise retry
        if json_response:
            break
        else:
            nth_generation += 1
            if nth_generation > kwargs.get('max_retry', 5):
                # Return error response with backup data if parse_response failed
                if 'parse_response' in [f.__name__ for f in post_processing_funcs]:
                    return {"fail_to_parse_response": secondary_response}

    return json_response

def print_json(data):
	logger.info(json.dumps(data, ensure_ascii=False, indent=2))

def save_json(data: List[Dict], file_path: str):
	with open(file_path, "w", encoding='utf-8') as f:
		json.dump(data, f, ensure_ascii=False, indent=2)

def read_json(file_path: str) -> List[Dict]:
	with open(file_path, 'r', encoding='utf-8') as f:
		data = json.load(f)
	return data

	
if __name__ == '__main__':
	messages = [{"role": "system", "content": "Hello, how are you? Hello, how are you? Hello, how are you?"}]
	model = 'gpt-4o'

	print(get_response(model, messages))
		
