# Transforming extracted book data to the training format and the test set
# The training data follows the sharegpt format for sft, such as : 
# [
#   [
#     {
#       "from": "Human",
#       "value": "Hello, how are you?"
#     },
#     {
#       "from": "Assistant",
#       "value": "I'm fine, thank you!"
#     }
#   ]
# ]

import os
import random
import argparse
from utils import remove_inner_thoughts
import json
import re
import copy

parser = argparse.ArgumentParser(description='Convert data format')
parser.add_argument('--dir', type=str, default='data', help='input_dir path for both input and output')
args = parser.parse_args()

input_dir = args.dir + '/final/'

# Get a list of all files in the input_dir
files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

def stable_shuffle(files):
    """
    Performs a deterministic shuffle of file names using a custom hash function.
    
    This function provides consistent ordering of files across different runs,
    unlike Python's built-in hash() which is randomized per process.
    
    Args:
        files: List of filenames (strings) to be shuffled
        
    Returns:
        List of filenames in a deterministically shuffled order
        
    Note:
        Uses a simple polynomial rolling hash function (similar to Java's String.hashCode())
        with a multiplier of 31 to generate consistent hash values.
    """
    def string_hash(s):
        """
        Computes a deterministic 32-bit hash value for a string.
        
        Args:
            s: Input string to hash
            
        Returns:
            32-bit integer hash value
        """
        h = 0
        # Multiply by 31 at each step to distribute bits well
        # Use bitwise AND with 0xFFFFFFFF to keep within 32 bits
        for c in s:
            h = (31 * h + ord(c)) & 0xFFFFFFFF
        return h
    
    # Generate (filename, hash) pairs for stable sorting
    files_with_hash = [(filename, string_hash(filename)) 
                       for filename in files]
    
    # Sort files based on their hash values for deterministic ordering
    files_with_hash.sort(key=lambda x: x[1])
    
    # Extract and return just the filenames in their new order
    return [file_hash_pair[0] for file_hash_pair in files_with_hash]

# Shuffle the files in a stable way
files = stable_shuffle(files)

# Print first 10 files and their total count
print(f"First 10 files: {files[:10]}")
print(f"Total number of files: {len(files)}")

dedup = False

ENVIRONMENT = 'Environment'
NSP = 'NSP'

def process(file):
    """
    This function process a book's extracted data file. Most importantly, it converts the data into GCA training data, which follow the sharegpt format for sft.  

    This function processes a JSON file containing book data and converts it into a format suitable
    for training LLMs. It handles:
    - Preparing Characters' System Prompts
    - Multiple processing modes (with/without inner thoughts, with/without other character profiles)
    - Next speaker prediction data generation
    - Weighting plots based on character frequency
    
    Args:
        file (str): Name of the JSON file containing extracted book data

    Returns:
        tuple: Contains:
            - train_chat_data (list): Chat-formatted training data
            - held_out_plots (list): The plots held-out for testing purposes, from which we sample test samples
            - character_profiles (dict): Character profile information
    """
    # Load and parse the input file
    with open(os.path.join(input_dir, file), 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    book_name = os.path.splitext(file)[0]

    # Extract character information
    character_datasets = data['character_datasets']
    character_profiles = {name: data['character_datasets'][name]['profile'] 
                         for name in character_datasets.keys()}

    import math 

    # Calculate character weights based on number of plot appearances
    character_weight = {name: max(1, math.sqrt(len(data['character_datasets'][name]['plots']))) 
                       for name in character_datasets.keys()}

    characters = character_datasets.keys()
    
    train_chat_data = []
    
    n_plots = len(data['plots'])

    # Default split is 90% train, 10% test unless specified in data
    split_index = data.get('split_plot_index', int(n_plots * 0.9))

    # Possible keys for message content
    message_keys = ['message', 'thought', 'description']

    # Process data in different modes for variety
    for thoughtless in [True, False]:  # With/without inner thoughts
        for with_other_character_profiles in [True, False]:  # With/without other character contexts
            for i_p, plot in enumerate(data['plots']):
                # Clean up plot data
                # plot.pop('text', None)

                # Standardize key_characters format
                plot_key_characters = [] 
                for c in plot['key_characters']:
                    if 'name' in c:
                        plot_key_characters.append(c)
                    elif 'character' in c:
                        c['name'] = c.pop('character')
                        plot_key_characters.append(c)
                
                plot['key_characters'] = plot_key_characters

                # Process each conversation in the plot
                for i_c, conversation in enumerate(plot['conversation']):
                    from collections import Counter
                    # Count utterances per character
                    utterance_counts = Counter(utterance['character'] 
                                            for utterance in conversation['dialogues'])
        
                    # Sort characters by frequency of speech
                    speaking_characters_w_env = sorted(utterance_counts, 
                                                     key=lambda x: utterance_counts[x], 
                                                     reverse=True)

                    plot_characters = [c['name'] for c in plot['key_characters']]

                    # Standardize conversation character format
                    conversation_key_characters = []
                    for c in conversation['key_characters']:
                        if 'name' in c:
                            conversation_key_characters.append(c)
                        elif 'character' in c:
                            c['name'] = c.pop('character')
                            conversation_key_characters.append(c)
                    
                    conversation['key_characters'] = conversation_key_characters

                    # Validate and standardize dialogue format
                    broken_sign = False
                    for i_u, utterance in enumerate(conversation['dialogues']):
                        if not 'message' in utterance:
                            if 'thought' in utterance:
                                utterance['message'] = utterance.pop('thought')
                            elif 'description' in utterance:
                                utterance['message'] = utterance.pop('description')
                            else:
                                # Check for bracketed messages
                                message = None
                                for k, v in utterance.items():
                                    if k.startswith('[') and k.endswith(']'):
                                        message = k + v
                                        break
                                if message is not None:
                                    utterance['message'] = message
                                    utterance.pop(k)
                                else:
                                    broken_sign = True
                                    break
                        
                        if not 'character' in utterance:
                            if 'name' in utterance:
                                utterance['character'] = utterance.pop('name')
                            else:
                                broken_sign = True
                                break
                    
                    if broken_sign:
                        continue

                    # Skip conversations with no valid characters
                    if len([c for c in utterance_counts if c != ENVIRONMENT]) == 0:
                        continue

                    # Identify major characters (those with profiles)
                    major_characters = [c for c in speaking_characters_w_env if c in character_profiles]
                    
                    if 'major_characters' not in conversation:
                        conversation['speaking_characters_w_env'] = speaking_characters_w_env
                        conversation['major_characters'] = major_characters
                    else:
                        assert conversation['speaking_characters_w_env'] == speaking_characters_w_env
                        assert conversation['major_characters'] == major_characters

                    # Build scenario string
                    if random.random() < 0.5:
                        scenario_str = conversation['scenario'] + '\n' + plot['summary']    
                    else:
                        scenario_str = conversation['scenario']

                    # Select characters for processing
                    if dedup:
                        # Select single character, either most frequent or random
                        if random.random() < 0.5:
                            selected_character = speaking_characters_w_env[0]
                        else:
                            count = 0
                            while True:
                                selected_character = random.choice(conversation['dialogues'])['character']
                                if selected_character in speaking_characters_w_env:
                                    break
                                count += 1
                        
                        selected_characters = [selected_character]
                    else:
                        selected_characters = speaking_characters_w_env

                    # Collect character profiles for this conversation
                    tmp_character_profiles = {}
                    for character in speaking_characters_w_env:
                        if character == ENVIRONMENT:
                            continue
                        character_profile = ''

                        if character in character_profiles:
                            character_profile = character_profiles[character]
                        
                        if character in plot_characters:
                            character_info = [c for c in plot['key_characters'] 
                                           if c.get('name', '') == character][0]

                            if 'description' in character_info:
                                character_profile = (character_info.get('description', '').strip('\n') 
                                                  + '\n\n' + character_profile.strip('\n'))
                        
                        character_profile = character_profile.strip(' \n')
                        if character_profile != '':
                            tmp_character_profiles[character] = character_profile

                    # Process each selected character
                    for character in selected_characters:
                        chat = []
                        
                        # Generate appropriate system prompt
                        if character == ENVIRONMENT:
                            from utils import get_environment_prompt
                            system_prompt = get_environment_prompt(
                                major_characters=major_characters,  
                                scenario=scenario_str
                            )
                        else:
                            # Get character motivation
                            character_info = [char for char in conversation['key_characters'] 
                                           if char.get('name', '') == character]

                            if len(character_info) == 0: 
                                motivation = ""
                            else:
                                try:
                                    motivation = character_info[0]['motivation']
                                except:
                                    motivation = character_info[0].get('description', '')

                                # 50% chance to drop the motivation
                                if random.random() < 0.5:
                                    motivation = ""

                            # Build character profile
                            character_profile = tmp_character_profiles.get(character, '')

                            if character in plot_characters:
                                character_info = [c for c in plot['key_characters'] 
                                               if c.get('name', '') == character][0]

                                if random.random() < 0.5:
                                    character_profile += '\n\n' + character_info.get('experience', '')
                            
                            character_profile = character_profile.strip(' \n')
                            
                            from utils import get_character_prompt
                            system_prompt = get_character_prompt(
                                book_name, character, character_profile, plot['summary'], conversation['scenario'], 
                                motivation, thoughtless, 
                                other_character_profiles=tmp_character_profiles if with_other_character_profiles else None
                            )

                        # Build chat sequence
                        chat.append({"from": "system", "value": system_prompt})
                        chat.append({"from": "user", "value": "===Conversation Start===\n\n"})
                        
                        prev_role = 'human'

                        import re

                        # Process each utterance
                        for i_u, utterance in enumerate(conversation['dialogues']):
                            message = utterance['message']

                            if utterance['character'] != character:
                                if utterance['character'] != ENVIRONMENT:
                                    message = remove_inner_thoughts(message)

                                if prev_role == 'human':
                                    chat[-1]['value'] += f"{utterance['character']}: {message}\n\n"
                                else:
                                    chat.append({"from": "user", 
                                               "value": f"{utterance['character']}: {message}\n\n"})
                                prev_role = 'human'
                            else:
                                if thoughtless:
                                    if utterance['character'] != ENVIRONMENT:
                                        message = remove_inner_thoughts(message)

                                    if len(message.strip(' \n')) == 0:
                                        continue
                                    
                                if prev_role == 'character':
                                    chat[-1]['value'] += f"{message}\n\n"
                                else:
                                    chat.append({"from": "assistant", 
                                               "value": f"{utterance['character']}: {message}\n\n"})
                                prev_role = 'character'
                        
                        # Skip if no assistant messages
                        if len([m for m in chat if m['from'] == 'assistant']) == 0:
                            continue 
                            
                        # Package chat data
                        chat = {
                            "conversations": chat,
                            "details": {
                                "book_name": book_name,
                                "plot_index": i_p,
                                "plot": plot,
                                "character": character
                            }
                        }
                        
                        # Add to appropriate dataset
                        if i_p < split_index:
                            train_chat_data.append(chat)
                        
                    
                    # Generate next speaker prediction data
                    from utils import get_nsp_prompt
                    chat = []
                    system_prompt = get_nsp_prompt(
                        all_characters=speaking_characters_w_env, 
                        scenario=scenario_str
                    )

                    chat.append({"from": "system", "value": system_prompt})
                    chat.append({"from": "user", "value": "===Conversation Start===\n\n"})

                    for i_u, utterance in enumerate(conversation['dialogues']):
                        next_actor = utterance['character']
                                        
                        # Randomly mask some non-major characters
                        if next_actor not in major_characters + [ENVIRONMENT]:
                            if random.random() < 0.25:
                                next_actor = "random"
                        elif next_actor not in speaking_characters_w_env:
                            next_actor = "random"
                                    
                        chat.append({"from": "assistant", "value": next_actor})

                        message = None
                        for key in message_keys:
                            if key in utterance:
                                message = utterance[key]
                                break
                        
                        if message is None:
                            break

                        chat.append({"from": "user", 
                                   "value": f"{utterance['character']}: {message}\n\n"})
                        
                    chat.append({"from": "assistant", "value": "<END CHAT>"})

                    chat = {"conversations": chat, "details": {
                        "book_name": book_name,
                        "plot_index": i_p,
                        "plot": plot,
                        "character": NSP
                    }}

                    # Add some NSP data to training set
                    if i_p < split_index and random.random() < 0.1:
                        train_chat_data.append(chat)
                        
    # Prepare plot-level dataset
    for i_p, plot in enumerate(data['plots']):
        plot['i_p'] = i_p

    def avg(lst):
        return sum(lst) / len(lst)

    held_out_plots = data['plots'][split_index:]
    # Weight each plot based on its characters
    for plot in held_out_plots:
        for conversation in plot['conversation']:
            try:
                conversation['weight'] = avg([
                    character_weight.get(c, 1) 
                    for c in conversation['speaking_characters_w_env'] 
                    if c != ENVIRONMENT
                ])
            except:
                conversation['weight'] = 1

    return train_chat_data, held_out_plots, character_profiles


# Process each file
n_books = len(files)

train_chat_data = []
held_out_plots_id_books = {}
held_out_plots_ood_books = {}


for i_b, file in enumerate(files):
    book_name = os.path.splitext(file)[0]
    train_chat_data_book,  test_plots_book, character_profiles_book = process(file)

    if i_b < n_books * 0.9: #and not ('A Song of Ice and Fire' in book_name) : # The training set excludes the GOT data 
        train_chat_data += train_chat_data_book

        held_out_plots_id_books[book_name] = {
            "character_profiles": character_profiles_book,
            "plots": test_plots_book,
        }
    else:
        held_out_plots_ood_books[book_name] = {
            "character_profiles": character_profiles_book,
            "plots": test_plots_book,
        }
    
# print key statistics of the datasets
print(f"n_books: {n_books}")
print(f"train_chat_data: {len(train_chat_data)}")


random.shuffle(train_chat_data)

os.makedirs(args.dir + '/train', exist_ok=True)
os.makedirs(args.dir + '/test', exist_ok=True)

# Save the chat data in the sharegpt format
def to_sharegpt_format(chat_data):
    results = []
    role_map = {
        'system': 'system',
        'user': 'human',
        'assistant': 'assistant',
    }
    for sample in chat_data:
        conversation = []
        for message in sample['conversations']:
            conversation.append({
                "from": role_map[message['from']],
                "value": message['value']
            })
        results.append({"conversations": conversation})
    return results

with open(args.dir + '/train/sft_sharegpt.json', 'w', encoding='utf-8') as f:
    sharegpt_format_data = to_sharegpt_format(train_chat_data)
    json.dump(sharegpt_format_data, f, ensure_ascii=False, indent=2)


def to_test_circumstance(test_plots, n_samples=100, tag=''):
    samples = []
    for book_name in test_plots.keys():
        for i, plot in enumerate(test_plots[book_name]['plots']):
            for j, conversation in enumerate(plot['conversation']):
                weight = conversation.get('weight', 1.0) 
                samples.append((book_name, i, j, weight))
    
    weights = [s[-1] for s in samples]

    if len(samples) < n_samples:
        print(f'In to_test_circumstance, Warning: there are only {len(samples)} samples, which is less than the expected number {n_samples}')
        n_samples = len(samples)

    # first select the top half of the samples
    samples = sorted(samples, key=lambda x: x[-1], reverse=True)
    samples = samples[:max(n_samples, len(samples) // 2)]

    weights = [s[-1] for s in samples]

    import random
    random.seed(42)

    import numpy as np
    np.random.seed(42)

    # Sample n_samples items according to weights
    selected_samples = []
    book_counts = {}

    weights = np.array(weights)
    weights = weights / weights.sum() # Normalize weights to probabilities
    
    while len(selected_samples) < n_samples:
        # Sample one index according to weights
        idx = np.random.choice(len(weights), p=weights)
        sample = samples[idx]
        book_name = sample[0]
        
        # Check if we can add this sample
        book_counts.setdefault(book_name, 0) 

        if sample in selected_samples: #or book_counts[book_name] >= 5:
            continue
        
        book_counts[book_name] += 1
        selected_samples.append(sample)
    
    samples = selected_samples

    assert(len(set(samples)) == len(samples))

    sampled_conversations = []
    # now arrange the data into a list 
    for book_name, i, j, weight in samples:
        plot = copy.deepcopy(test_plots[book_name]['plots'][i])
        conversation = copy.deepcopy(plot['conversation'][j])
        plot.pop('conversation', None)
        
        for _ in conversation['key_characters']:
            _.pop('i_p', None)
            _.pop('i_c', None)
        
        for _ in conversation['dialogues']:
            _.pop('i_p', None)
            _.pop('i_c', None)
            _.pop('i_u', None)
        
        for _ in plot['key_characters']:
            _.pop('i_p', None)

        conversation['plot'] = plot
        conversation['character_profiles'] = { c: test_plots[book_name]['character_profiles'][c] for c in conversation['major_characters'] }
        conversation['book'] = book_name
        conversation['i_p'] = plot['i_p']
        conversation['i_c'] = j
        conversation['tag'] = tag

        sampled_conversations.append(conversation)
        print(f'Test Set Sample {book_name} {i} {j}')

    return sampled_conversations

# Save the test plots
with open(args.dir + '/test/held_out_plots.json', 'w', encoding='utf-8') as f:
    json.dump({'id': held_out_plots_id_books, 'ood': held_out_plots_ood_books}, f, ensure_ascii=False, indent=2)

test_circumstance_id = to_test_circumstance(held_out_plots_id_books, n_samples=100, tag='id')
test_circumstance_ood = to_test_circumstance(held_out_plots_ood_books, n_samples=100, tag='ood')

with open(args.dir + '/test/test_set.json', 'w', encoding='utf-8') as f:
    test_circumstance = test_circumstance_id + test_circumstance_ood
    json.dump(test_circumstance, f, ensure_ascii=False, indent=2)



