# Standard library imports
import argparse
import json
import os
import re
import traceback
from collections import Counter
from typing import List, Tuple
# Third-party imports
import jsonlines
from tqdm import tqdm
import tiktoken
import re
import difflib
# Local imports
from utils import config, cached, get_response, setup_logger, get_response_json, print_json, encode, decode

def parse_args():
    parser = argparse.ArgumentParser(description='Construct CoSER-style dataset from source books')
    parser.add_argument('--input', type=str, required=True,
                      help='Input jsonl file path containing books data')
    parser.add_argument('--output_dir', type=str, default='data',
                      help='Output directory path (default: data')
    parser.add_argument('--num_workers', type=int, default=1,
                      help='Number of parallel workers (default: 1)')
    parser.add_argument('--model', type=str, default="gpt-4o",
                      help='Model to use for data construction (default: gpt-4o)')
    parser.add_argument('--candidate_model', type=str, default="gpt-4o",
                      help='Another candidate model to use for data construction when the main model fails (default: gpt-4o)')
    parser.add_argument('--regenerate', action='store_true',
                      help='Force regenerate data even if results already exist (default: False)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    return args

args = parse_args()

# Setup logger
logger = setup_logger(__name__, f'{__file__.split(".")[0]}.log')

def find_index(lst, key):
    """
    Find the index of a key in a list, returning -1 if not found.

    Args:
        lst: The list to search in
        key: The key to search for

    Returns:
        int: The index of the key if found, -1 if not found
    """
    try:
        return lst.index(key)
    except ValueError:
        return -1

@cached
def create_chunk_generator(book, chunk_size):
    """
    Generates chunks of text from a book while respecting token limits and chapter boundaries.

    Args:
        book (dict): A dictionary containing book information with 'content' and other fields
        chunk_size (int): Roughly the number of tokens per chunk

    Returns:
        list: A list of text chunks from the book, where each chunk is:
            - Limited to chunk_size if no chapters are detected
            - Between chunk_size/2 and 2*chunk_size if chapters are detected
            - Cleaned of copyright notices in the first chunk
            - Cleaned of excessive tabs if present

    The function handles books in two ways:
    1. For books without chapter markers: Splits into fixed-size chunks of chunk_size
    2. For books with chapters: Attempts to keep chapters together while staying within token limits
    """
    # Check and clean excessive tabs that may interfere with text processing
    def has_excessive_tabs(content, threshold=0.05):
        tab_count = content.count('\t')
        return (tab_count / len(content)) > threshold
    
    if has_excessive_tabs(book['content']):
        book['content'] = book['content'].replace('\t', '')

    # Try to split book into chapters using split_book utility
    from split import split_book
    chapters = split_book(book)

    results = []

    # Case 1: No chapters detected - split into fixed-size chunks
    if not chapters:
        # Convert text to tokens for more precise chunking
        tokens = encode(book['content'])
        start_index = 0
        
        while start_index < len(tokens):
            # Handle the last chunk which may be smaller than chunk_size
            if len(tokens) - start_index <= chunk_size:
                chunk = decode(tokens[start_index:])
                results.append(chunk)
                break
            
            # Create chunk of specified size and decode back to text
            chunk_tokens = tokens[start_index:start_index + chunk_size]
            chunk = decode(chunk_tokens)
            results.append(chunk)
            
            start_index += len(chunk_tokens)
    
    # Case 2: Chapters detected - try to keep chapters together while respecting size limits
    else:
        current_chunk = []
        current_tokens = 0
        
        for chapter in chapters:
            # Add chapter to current chunk
            current_chunk.append(chapter['content'])
            current_tokens += len(encode(chapter['content']))
            
            # Check if current chunk has reached minimum size (chunk_size/2)
            if current_tokens >= chunk_size // 2:
                # If chunk is within acceptable size range (between chunk_size/2 and 2*chunk_size)
                if current_tokens <= 2 * chunk_size:
                    results.append(''.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                # If chunk is too large, split it into smaller pieces
                else:
                    chunk_text = ''.join(current_chunk)
                    chunk_tokens = encode(chunk_text)
                    for i in range(0, len(chunk_tokens), chunk_size):
                        # Special handling for final piece to avoid tiny chunks
                        if i + 2 * chunk_size >= len(chunk_tokens):
                            results.append(decode(chunk_tokens[i:]))
                            break
                        else:
                            results.append(decode(chunk_tokens[i:i + chunk_size]))
                    current_chunk = []
                    current_tokens = 0
        
        # Add any remaining content as final chunk
        if current_chunk:
            results.append(''.join(current_chunk))

    # Clean copyright notices from the first chunk to avoid irrelevant text
    lines = results[0].split('\n')
    filtered_lines = []
    copyright_words = ['rights', 'reserved', 'reproduced', 'copyright', 'reproduce', 'permission']
    
    # Remove lines that are likely copyright notices (short lines with multiple copyright-related words)
    for line in lines:
        words = line.split()
        if len(words) < 50 and sum(word.lower() in copyright_words for word in words) > 1:
            continue
        filtered_lines.append(line)
    
    results[0] = '\n'.join(filtered_lines)

    return results


def ngram_jaccard_similarity(text1, text2, n=3):
    """Calculate the Jaccard similarity between two texts using n-grams.
    
    Args:
        text1 (str): First text to compare
        text2 (str): Second text to compare 
        n (int, optional): Size of n-grams. Defaults to 3.
    
    Returns:
        float: Jaccard similarity score between 0 and 1, where 1 means identical texts
              and 0 means completely different texts.
    """
    def ngrams(tokens, n):
        """Generate n-grams from a sequence of tokens.
        
        Args:
            tokens (list): List of tokens
            n (int): Size of n-grams
        Returns:
            list: List of n-gram tuples
        """
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def jaccard_similarity(set1, set2):
        """Calculate Jaccard similarity between two sets.
        
        Args:
            set1 (set): First set
            set2 (set): Second set
        Returns:
            float: Jaccard similarity score
        """
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0

    # Tokenize the input texts into sequences of tokens
    tokens1 = encode(text1)
    tokens2 = encode(text2)
    
    # Generate sets of n-grams from the token sequences
    ngrams1 = set(ngrams(tokens1, n))
    ngrams2 = set(ngrams(tokens2, n))
    
    # Calculate and return the Jaccard similarity between the n-gram sets
    return jaccard_similarity(ngrams1, ngrams2)

@cached
def find_best_match_passage(candidates, target, n=3, threshold=0.3):
    """Find the best matching passage from a list of candidates compared to a target text. These texts are generally LLM-synthesized summaries. Hence, we focus on their semantic similarity.

    Uses n-gram Jaccard similarity to compare texts and find the closest match.
    
    Args:
        candidates (list): List of candidate passages to search through
        target (str or dict): Target text to match against
        n (int, optional): Size of n-grams to use for comparison. Defaults to 3.
        threshold (float, optional): Minimum similarity score required to consider a match.
                                   Defaults to 0.3.
    
    Returns:
        int: Index of best matching passage if score >= threshold, -1 if no good match found
    """
    best_match = None  # Index of current best matching passage
    best_score = 0     # Highest similarity score found so far

    # Handle case where inputs are dictionaries by converting to strings
    if isinstance(candidates, list) and isinstance(target, dict) and isinstance(candidates[0], dict):
        target = str(target)
        candidates = [str(c) for c in candidates]

    # Compare target against each candidate passage
    for i, candidate in enumerate(candidates):
        score = ngram_jaccard_similarity(target, candidate, n)
        if score >= best_score:
            best_score = score
            best_match = i
    
    # Return best match if it meets threshold, otherwise return -1
    if best_score >= threshold:
        logger.debug(f"Best match: \nInput: {target}\nOutput: {candidates[best_match]}\nScore: {best_score}")
        return best_match
    else:
        return -1


@cached
def find_best_match_sentence(chunk, target, threshold=0.6):
    """Find the best matching sentence from a chunk of text or list of sentences compared to a target sentence. These sentences are generally exact sentences from the book, so we focus on their string similarity.
    
    Uses SequenceMatcher to calculate string similarity ratios between sentences to find the closest match.
    
    Args:
        chunk (str or list): Input text chunk or list of sentences to search through
        target (str): Target sentence to match against
        threshold (float, optional): Minimum similarity score required to consider a match.
                                   Defaults to 0.6.
    
    Returns:
        str or None: Best matching sentence if score >= threshold, None if no good match found
                    or if target is None/invalid
    """
    # Return None for invalid target inputs
    if target == 'None' or target is None:
        return None

    # Split chunk into sentences if it's a string, otherwise use as-is if it's a list
    if isinstance(chunk, str):
        # Split on sentence endings followed by whitespace
        sentences = re.split(r'(?<=[.!?。！？\n])\s*', chunk)
    else: # chunk is a list
        assert isinstance(chunk, list)  
        sentences = chunk

    # Initialize variables to track best match
    best_match = 0
    best_score = 0
    
    # Compare target against each sentence
    for i, sentence in enumerate(sentences):
        # Calculate similarity ratio between target and current sentence
        score = difflib.SequenceMatcher(None, target, sentence).ratio()
        
        # Update best match if current score is higher
        if score > best_score:
            best_score = score
            best_match = sentence
    
    # Log the matching results
    logger.debug(f"Best match: \nInput: {target}\nOutput: {best_match}\nScore: {best_score}")

    # Return best match if it meets threshold, otherwise return None
    if best_score >= threshold:
        return best_match
    else:
        return None
def extract_from_chunk(book, i_c, chunk, truncated_plots=None):
    """
    Extract and process plot information from a chunk of book text.
    
    This function analyzes a chunk of text to identify chapter beginnings, plots, conversations,
    and other narrative elements. It uses an LLM to generate structured information about the text.

    Args:
        book (dict): Dictionary containing book metadata including title and author
        i_c (int): Chunk index
        chunk (str): Text content of the current chunk to analyze
        truncated_plots (list, optional): List of incomplete plots from previous chunk that need to be finished

    Returns:
        tuple: Contains:
            - chapter_beginnings (list): List of identified chapter starts
            - plots (list): Extracted plot information including summaries, characters, conversations
            - remaining_chunk (str): Unused portion of chunk to process in next iteration
            
    The function generates a detailed prompt for the LLM that requests:
    1. Chapter beginning identification
    2. Plot extraction and analysis
    3. Conversation reconstruction
    4. Character motivation analysis
    5. Next chunk starting point determination
    """
    logger.info(f"Extracting plots from chunk for book: {book['title']}")

    # Create deep copy of truncated plots and remove text field to avoid redundancy
    import copy
    if truncated_plots:
        truncated_plots = copy.deepcopy(truncated_plots)
        for plot in truncated_plots:
            plot.pop('text')
    
    # Construct the prompt for the LLM
    prompt = f"""
Based on the provided book chunk, complete the following tasks:

1. Recognize chapter beginnings if they exist in the chunk. Identify the starting sentence of that chapter.
2. Identify the important plots in this chunk. Identify the beginning and ending of each plot by its first and last sentence. Determine the chapter title that the plot belongs to. Set "state" as "truncated" if the plot is truncated in this chunk, or "finished" otherwise. You will be provided with the truncated plots from the previous chunk, and you **must** extend the conversations with the current chunk while keeping the **scenario** unchanged. 
3. Summarize each important plot. For each plot, generate its summary, score its prominence from 1 to 100, and list the key characters and their roles, thoughts and actions in it.
4. Extract conversations for each plot. First, state the scenario and topic of the conversations. Then, list the key characters with their names, descriptions and thoughts (motivations) at this point. Finally, extract the conversations among them based on the following requirements: 
    i) Ensure the conversations are faithful to the plot and characters. They should be based on the original conversations in the text as much as possible. 
    ii) The conversations should be complete, covering the key dialogues and information. Each conversation should contain at least 10 utterances.
    iii) Each utterance should be composed of one or more thoughts, speech and actions. Use [] outside thoughts, like "[I feel fear and anger, but I cannot show it. I must remain calm and carefully handle his volatile temper.]", which others can't see. Use () outside actions, like "(silence)" or "(smiles at you)," which others can see. Always start an utterance with the character's thought. 
    iv) [IMPORTANT] When generating thoughts, you should think from the characters' perspectives, analyzing the internal thoughts behind their speech and actions in the original text. These thoughts should reflect aspects such as their personal background, personality, values, relationships with others, motivations, and goals. Each thought should be expressed as a phrase or sentence, rather than an adjective or adverb. 
    v) Additionally, describe environmental information (such as scenes, atmosphere, sudden events, etc.) of the conversations as an "utterance" where the "character" field is set as "Environment". The information should exclude characters' active thoughts, observations, and actions.
    vi) Keep the conversation in the same language as the chunk. 
5. Identify the optimal starting point for the subsequent chunk. If the last storyline has been extracted as an truncated plot, set next_chunk_start as None. Otherwise, set next_chunk_start as the first sentence of the last storyline. 

===Output Format===
Please provide the output in the following JSON format:
{{
    "chapter_beginnings": [
        {{
            "beginning_sentence": "Exactly the first line of this chapter (namely the title)."
        }}
    ],
    "plots": [
        // Extend the truncated plots from previous chunk, if any
        {{
            ...
        }}, 
        // New plots in this chunk
        {{
            "chapter_title": "The chapter title that the plot belongs to. Output None if not found.",
            "first_sentence": "Exactly the first sentence of the plot in this **chunk**.",
            "last_sentence": "Exactly the last sentence of the plot in this **chunk**. If the plot is truncated in this chunk, provide the last sentence of this chunk. ",
            "prominence": "Whether this plot is recognized to fans of this book, from 1 to 100.",
            "summary": "The summary of the plot. Just summarize, do not extend unrelated discussions.",
            "key_characters": [
                {{
                    "name": "Character name",
                    "description": "The description of the character before this plot (~20 words).",
                    "experience": "The summary of the character's role, thoughts and behaviors towards this plot, and any significant character development relevant to the plot (~30 words).",
                }}
            ],
            "conversation": [{{
                "scenario": "The scenario at the start of this conversation (providing as much context as possible, but excluding details conveyed in the following conversation)",
                "topic": "The topic of the conversation (~10 words)", 
                "key_characters": [
                    {{
                        "name": "Character name",
                        "motivation": "The thought of the character before starting the conversation, including their attitudes, feelings, motivations, goals, information to convey or topics to be discussed",
                    }}
                ],
                "dialogues": [
                    {{
                        "character": "Character name",
                        "message": "Message, each utterance is composed of thoughts, speech and actions. Use [thought] for internal thoughts, like "[feeling happy]", which others can't see. Use (action) for visible actions, like "(silence)" or "(smiles at you)". Each response starts with the character's internal thought before their speech and actions."
                    }}
                ]
            }}],
            "state": "finished" or "truncated"
        }}
    ],
    "next_chunk_start": "The first sentence of the next chunk."
}}

===Requirements===
1. Adhere strictly to the specified output JSON format. 
2. [IMPORTANT] Ensure all DOUBLE QUOTES within all STRINGS are properly ESCAPED, especially when extracting from the text.
3. In the OUTPUT, use characters' full names, omitting any titles.
4. Maintain Story Fidelity: The plot must accurately reflect the book's content. Avoid introducing plots that are out of context. If the plot contains multiple conversations, prioritize the original dialogue from the book. In the absence of explicit conversations, create dialogue that aligns closely with the plot details.

===Input===

==Book title==
{book['title']}

==Author==
{book['author']}

==Chunk of Book Content== 
{chunk}

==Truncated plot from previous chunk (to be finished)==
{json.dumps(truncated_plots, ensure_ascii=False, indent=2) if truncated_plots else "None"}
"""
    
    # Example format for character utterances in conversations
    # "[My father's words fill me with awe, but I still feel uneasy.] 
    # (Nods seriously, but with a slight frown remaining) 
    # I understand, Father. Responsibility is important. But… is killing really necessary? 
    # (A flash of compassion in his eyes)
    # If someone has done something wrong, can't we give them a chance to make amends?"

    logger.debug(prompt)

    def parse_response(response, chunk, book, **kwargs):
        """
        Parse and validate the LLM response, extracting structured plot information.
        
        Args:
            response: Raw LLM response to parse
            chunk: Original text chunk for reference
            book: Book metadata
            **kwargs: Additional keyword arguments
            
        Returns:
            tuple or bool: (chapter_beginnings, plots, remaining_chunk) if successful, False if failed
        """
        if not response:
            return False
        
        try:
            # Handle different response formats
            # Sometimes response is just a list of plots
            if (isinstance(response, dict) and 'first_sentence' in response):
                response = [response]
            elif isinstance(response, list) and 'first_sentence' in response[0]:
                response = {
                    'chapter_beginnings': [],
                    'plots': response,
                    'next_chunk_start': None
                }

            try:
                chapter_beginnings = response['chapter_beginnings']
            except:
                print(f"Error: {response}")

            plots = []

            # Process next chunk starting point
            if response['next_chunk_start']:
                response['next_chunk_start'] = find_best_match_sentence(chunk, response['next_chunk_start'])

                # Calculate remaining chunk (20% of original)
                remaining_chunk = chunk[chunk.index(response['next_chunk_start']):]
                remaining_chunk = remaining_chunk[int(len(remaining_chunk) * 0.2):]
                # Ensure remaining chunk starts with complete sentence
                remaining_chunk = remaining_chunk[find_index(remaining_chunk, '\n') + 1:] 
            else:
                remaining_chunk = ''
            
            # Process each plot from the response
            for unprocessed_plot in response['plots']:
                
                chapter_title = unprocessed_plot['chapter_title']

                # Find exact matches for plot boundaries in original text
                unprocessed_plot['first_sentence'] = find_best_match_sentence(chunk, unprocessed_plot['first_sentence'])
                unprocessed_plot['last_sentence'] = find_best_match_sentence(chunk, unprocessed_plot['last_sentence'])

                first_sentence, last_sentence = unprocessed_plot['first_sentence'], unprocessed_plot['last_sentence']

                # Extract original text for this plot
                original_text = chunk[chunk.index(first_sentence):chunk.index(last_sentence) + len(last_sentence)]

                # Create structured plot object
                plot = {
                    'text': original_text,
                    'summary': unprocessed_plot['summary'],
                    'prominence': unprocessed_plot['prominence'],
                    'key_characters': unprocessed_plot['key_characters'],
                    'chapter': chapter_title,
                    'conversation': unprocessed_plot['conversation'],
                    'state': unprocessed_plot['state']
                }

                plots.append(plot)

            # Log processed response
            print_json(response)
            logger.info(json.dumps(response, ensure_ascii=False, indent=2))

            return chapter_beginnings, plots, remaining_chunk
        
        except Exception as e:
            logger.error(f"Error processing chunk for book {book['title']}: {e}, {traceback.format_exc()}")
            return False
    
    from utils import get_response_json, extract_json

    # Get and parse LLM response
    response = get_response_json([extract_json, parse_response], model=args.model, messages=[{"role": "user", "content": prompt}], book=book, chunk=chunk, fix_truncated_json=True)

    return response

def extract(book, chunk_size=8192):
    """Process a book by splitting it into chunks and extracting structured information.

    This function processes a book by:
    1. Splitting the book text into chunks of specified size
    2. Extracting chapter beginnings, plots and conversations from each chunk
    3. Handling truncated plots that span multiple chunks by merging them
    4. Saving the extracted results to a JSON file

    Args:
        book (dict): Book data containing 'title', 'author', and 'content'
        chunk_size (int, optional): Roughly the number of tokens per chunk. Defaults to 8192.

    Returns:
        dict: Extracted results containing:
            - chapter_beginnings: List of chapter names (start locations)
            - plots: List of extracted plots with conversations
            - fail_to_parse_responses: List of chunks that failed parsing
    """
    # Set up save path and skip if already processed
    save_dir = f'{args.output_dir}/extracted'
    os.makedirs(save_dir, exist_ok=True)

    save_path = f'{save_dir}/{book["title"]}.json'
    if os.path.exists(save_path) and not args.regenerate:
        return 

    # Set up cache path
    from utils import set_cache_path
    set_cache_path(f'.cache/{book["title"]}.pkl')

    # Create generator to iterate through book chunks
    chunk_generator = create_chunk_generator(book, chunk_size)

    # Initialize results structure
    results = {
        'chapter_beginnings': [],
        'plots': [],
    }

    # Track state between chunks
    remaining_chunk = ''  # Text carried over from previous chunk
    truncated_plots = []  # Plots that continue into next chunk
    
    # Process each chunk
    for i, chunk in enumerate(chunk_generator):
        
        print(f"Processing chunk {i} with {len(encode(chunk))} tokens")

        # Extract information from current chunk
        response = extract_from_chunk(book, i, remaining_chunk + chunk, truncated_plots)

        fail_to_parse_responses = []

        # Handle the response
        if response:
            if isinstance(response, tuple) and len(response) == 3:
                # Successful extraction
                chapter_beginnings, plots, remaining_chunk = response 
            else:
                # Failed extraction
                chapter_beginnings, plots, remaining_chunk = [], [], ''
                fail_to_parse_responses.append(response['fail_to_parse_response'])
        else:
            # No response
            chapter_beginnings, plots, remaining_chunk = [], [], ''

        # Merge truncated plots from previous chunk with current plots
        for u_plot in truncated_plots:
            # Find matching plot in current chunk (by summary similarity)
            idx = find_best_match_passage([p['summary'] for p in plots], u_plot['summary'])

            if idx != -1:
                # Found matching plot - merge them
                plots[idx]['text'] = u_plot['text'] + plots[idx]['text']
  
                # Merge conversations
                old_conversations = u_plot['conversation']
                new_conversations = plots[idx]['conversation']
                merged_conversations = []

                # Check each previous conversation
                for prev_conv in old_conversations:
                    idx_c = find_best_match_passage([s['scenario'] for s in new_conversations], prev_conv['scenario'])

                    if idx_c != -1:
                        # Use new conversation if scenarios match
                        merged_conversations.append(new_conversations[idx_c])
                    else:
                        # Keep old conversation if no match
                        merged_conversations.append(prev_conv)
                
                # Add any new conversations not already merged
                merged_conversations += [ c for c in new_conversations if c not in merged_conversations ]
                plots[idx]['conversation'] = merged_conversations
            else:
                # No matching plot found - mark as finished
                u_plot['state'] = 'finished'
                results['plots'].append(u_plot)

        # Separate/Update finished and truncated plots
        finished_plots = [plot for plot in plots if plot['state'] == 'finished']
        truncated_plots = [plot for plot in plots if plot['state'] == 'truncated']

        # Add to results
        results['chapter_beginnings'].extend(chapter_beginnings)
        results['plots'].extend(finished_plots)

    # Finish any remaining truncated plots
    for u_plot in truncated_plots:
        u_plot['state'] = 'finished'
        results['plots'].append(u_plot)
    
    results['fail_to_parse_responses'] = fail_to_parse_responses
    
    # Save results
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results

count_nth_generation = {i: 0 for i in range(7)}

def restore_from_cache(book):
    """
    As we typically encounter issues during the extraction process, this function restores some previously extracted plot data from cache.
    
    This function loads cached responses from extraction LLMs, processes them into the regular data format, and merges them with the results of extract(). 

    Args:
        book (dict): Book data containing title, author, and content

    Returns:
        None: Results are saved to disk, no return value
    """
    # Load existing extracted results

    save_dir = f'{args.output_dir}/extracted'
    os.makedirs(save_dir, exist_ok=True)

    with open(f'{save_dir}/{book["title"]}.json', 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    save_path = f'{save_dir}/{book["title"]}.json'

    # Skip if already processed
    if os.path.exists(save_path) and not args.regenerate:
       return 

    # Load cached API responses
    import pickle
    with open(f'.cache/cache_{book["title"]}.pkl', 'rb') as f:
        cache = pickle.load(f)
    
    # Get only the get_response cache entries
    keys = [ k for k in cache.keys() if k[0] == 'get_response' ]

    global count_nth_generation

    fail_prompts = []
    responses = {}

    # Generate chunks from book content
    chunk_generator = create_chunk_generator(book, chunk_size=8192)
    chunks = [chunk for chunk in chunk_generator]

    # Process each cached response
    for key, value in cache.items():
        if key[0] == 'get_response':
            # Extract kwargs from cache key
            dict_string = key[-1][11:-1]
            import ast
            parsed_list = ast.literal_eval(dict_string)
            restored_kwargs = dict(parsed_list)

            # Only process responses for plot extraction prompts
            if restored_kwargs['model'] == 'claude-3-5-sonnet-20240620' and restored_kwargs['messages'][0]['content'].startswith("\nBased on the provided book chunk, complete the following tasks:\n\n1. Recognize chapter beginnings if"):
                # Verify book title matches
                if not restored_kwargs['book']['title'] == book['title']:
                    logger.info(f"Warning: {restored_kwargs['book']['title']} != {book['title']}")
                    continue

                # Track generation attempts
                nth_generation = restored_kwargs['nth_generation']
                count_nth_generation[nth_generation] += 1

                # Store response
                prompt = restored_kwargs['messages'][0]['content']
                responses.setdefault(prompt, {})
                responses[prompt][nth_generation] = value

                # Track failed prompts (those that needed max retries)
                if nth_generation == 5:
                    fail_prompts.append(prompt)

    fetched_plots = []

    # Process failed prompts to extract any valid plots
    for prompt in fail_prompts:
        for nth_generation in range(6):
            if nth_generation in responses[prompt]:
                response = responses[prompt][nth_generation]
                
                # Check if response contains all required fields
                required_fields = ["chapter_beginnings", "plots", "chapter_title", "first_sentence", "last_sentence", "summary", "key_characters", "name", "description", "dialogues", "message"]
                if all(field in str(response) for field in required_fields):
                    # Extract JSON from potentially truncated response
                    from utils import extract_json
                    response = extract_json(response, post_fix_truncated_json=True)

                    if response is None:
                        continue

                    # Helper function to parse response and extract plots
                    def parse_response(response, chunk, book, **kwargs):
                        if not response:
                            return False
                        
                        try:
                            # Normalize response format
                            if (isinstance(response, dict) and 'first_sentence' in response):
                                response = [response]
                            elif isinstance(response, list) and 'first_sentence' in response[0]:
                                response = {
                                    'chapter_beginnings': [],
                                    'plots': response,
                                    'next_chunk_start': None
                                }

                            try:
                                chapter_beginnings = response['chapter_beginnings']
                            except:
                                print(f"Error: {response}")

                            plots = []

                            # Handle remaining chunk logic
                            if response['next_chunk_start']:
                                response['next_chunk_start'] = find_best_match_sentence(chunk, response['next_chunk_start'])
                                if response['next_chunk_start']:
                                    remaining_chunk = chunk[chunk.index(response['next_chunk_start']):]
                                    # Keep up to 20% content of the chunk
                                    remaining_chunk = remaining_chunk[int(len(remaining_chunk) * 0.2):]
                                    # Ensure remaining chunk starts with a sentence
                                    remaining_chunk = remaining_chunk[find_index(remaining_chunk, '\n') + 1:]
                                else:
                                    remaining_chunk = ''
                            else:
                                remaining_chunk = ''
                            
                            # Process each plot in the response
                            for unprocessed_plot in response['plots']:
                                chapter_title = unprocessed_plot['chapter_title']

                                # Match first and last sentences
                                unprocessed_plot['first_sentence'] = find_best_match_sentence(chunk, unprocessed_plot['first_sentence'], threshold=0.6)
                                unprocessed_plot['last_sentence'] = find_best_match_sentence(chunk, unprocessed_plot['last_sentence'], threshold=0.6)

                                first_sentence, last_sentence = unprocessed_plot['first_sentence'], unprocessed_plot['last_sentence']

                                # Extract original text if sentences found
                                if unprocessed_plot['first_sentence'] and unprocessed_plot['last_sentence']:
                                    original_text = chunk[chunk.index(first_sentence):chunk.index(last_sentence) + len(last_sentence)]
                                else:
                                    original_text = ''

                                # Create plot object
                                plot = {
                                    'text': original_text,
                                    'summary': unprocessed_plot['summary'],
                                    'prominence': unprocessed_plot['prominence'],
                                    'key_characters': unprocessed_plot['key_characters'],
                                    'chapter': chapter_title,
                                    'conversation': unprocessed_plot['conversation'],
                                    'state': unprocessed_plot['state']
                                }

                                plots.append(plot)

                            print_json(response)
                            logger.info(json.dumps(response, ensure_ascii=False, indent=2))

                            return chapter_beginnings, plots, remaining_chunk
                        
                        except Exception as e:
                            logger.error(f"Error processing chunk for book {book['title']}: {e}, {traceback.format_exc()}")
                            return False
                    
                    # Extract chunk from prompt
                    chunk = prompt.split('==Truncated plot from previous chunk (to be finished)==')[0].split('==Chunk of Book Content==')[-1].strip(' \n')

                    # Parse response to get plots
                    res = parse_response(response, chunk, book)

                    if res :
                        chapter_beginnings, plots, remaining_chunk = res
                    else:
                        continue

                    # Process extracted plots
                    for plot in plots:
                        plot['state'] = 'finished'
                        plot['i_chunk'] = -1
                        # Find which chunk this plot belongs to
                        for i_chunk, another_chunk in enumerate(chunks):
                            if another_chunk.strip(' \n').endswith(chunk[-100:]):
                                plot['i_chunk'] = i_chunk
                                break

                    fetched_plots.extend(plots)
                    break 
    
    # Find chunk indices for original plots
    for plot in results['plots']:
        plot['i_chunk'] = -1
        for i_chunk, chunk in enumerate(chunks):
            if plot['text'][-100:] in chunk:
                plot['i_chunk'] = i_chunk
                break

    # Merge and sort all plots
    logger.info(f'Number of Original Plots: {len(results["plots"])}, Fetched New Plots: {len(fetched_plots)}, Total Plots: {len(results["plots"]) + len(fetched_plots)}')

    new_plots = results['plots'] + fetched_plots
    new_plots = sorted(new_plots, key=lambda x: x['i_chunk'])

    results['plots'] = new_plots

    # Save restored results (together with the original results)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return 

def assemble(book):
    """
    Assemble the extracted plot data for a book into the final structured data.
    
    This function:
    1. Processes extracted plots and conversations
    2. Enhances conversation scenarios and character motivations using LLM
    3. Normalizes and standardizes character names
    4. Generates character profiles and datasets
    5. Saves the final assembled data
    
    Args:
        book (dict): Book data containing title, author and content
        
    Returns:
        None: Results are saved to disk at {args.output_dir}/final/{book_title}.json
    """
    # Set up caching for this book
    from utils import set_cache_path
    set_cache_path(f'.cache/cache_{book["title"]}.pkl')
    
    os.makedirs(f'{args.output_dir}/final', exist_ok=True)

    save_path = f'{args.output_dir}/final/{book["title"]}.json'

    # Skip if already processed
    if os.path.exists(save_path) and not args.regenerate:
        return 
    
    logger.info(f"Assembling book: {book['title']}")
    
    # Load extracted plot data
    with open(f'{args.output_dir}/extracted/{book["title"]}.json', 'r', encoding='utf-8') as f:
        results = json.load(f)

    plots = results['plots']

    # Detect language from first plot text
    if len(plots) > 0:
        from utils import lang_detect
        language = lang_detect(plots[0]['text'][:100])
        language = {'zh': 'Chinese', 'en': 'English'}.get(language, 'English')
    else:
        language = 'English'

    # Normalize conversation format and enhance scenarios/motivations
    for plot in plots:
        # Convert single conversation dict to list format
        if isinstance(plot['conversation'], dict) and 'scenario' in plot['conversation']:
            plot['conversation'] = [plot['conversation']]

        if plot['conversation'] is None:
            continue 
        for conversation in plot['conversation']:
            # Prepare input for conversation enhancement
            input_conversation = {
                'plot_summary': plot['summary'], 
                'character_information': plot['key_characters'],
                **conversation
            }
            
            # Get character names for this conversation
            conv_key_characters = [
                _.get('name', _.get('character', '')) 
                for _ in conversation['key_characters'] 
                if 'name' in _ or 'character' in _
            ]

            # Generate prompt for enhancing scenario and motivations
            prompt = f"""
Given a conversation from {book['title']}, enhance the scene setup and characters' motivations to create a comprehensive foundation for dramatic performance, i.e., to provide necessary background for actors to act out the conversation:

1. Review the provided conversation and contextual details thoroughly.
2. Expand the 'scenario' with rich situational context that actors need to convincingly perform the scene. Focus on essential background information, while excluding future details to be portrayed in the conversation.
3. Enhance each character's 'motivation' section with their complete mental and emotional state, including their feelings, ideas, objectives, topics they want to discuss, and information they want to convey. Align with their established character and role in the plot. 

===Output Format===
Please provide the output in the following JSON format:
{{
    "scenario": "A detailed scene-setting description that provides actors with essential context and atmosphere (< 200 words). Include all necessary background information while excluding future information to be revealed in the conversation.",
    "key_characters": [
        {{
            "name": "Character name",
            "motivation": "The character's complete mental and emotional state before the conversation (< 100 words). Including their feelings, motivations, objectives, and information they want to convey or discuss."
        }}
    ],
}}

===Requirements===
1. Adhere strictly to the specified output JSON format. 
2. [IMPORTANT] Ensure all DOUBLE QUOTES within all STRINGS are properly ESCAPED, especially when extracting from the text.
3. In the OUTPUT, keep the character names unchanged. 
4. Output in the same language as the input. 
5. Ensure the key_characters include exactly the same characters as the input, including {conv_key_characters}.

===Input Conversation and Background===
{json.dumps(input_conversation, ensure_ascii=False, indent=2)}
"""

            # Helper function to validate enhanced conversation response
            from utils import extract_json
            def parse_response(response, characters, **kwargs):
                try:
                    assert 'scenario' in response 
                    assert 'key_characters' in response
                    key_characters = {_['name']: _['motivation'] for _ in response['key_characters']}
                    for character in characters:
                        assert character in key_characters
                    return response
                except:
                    return False

            # Get enhanced conversation from LLM
            response = get_response_json(
                [extract_json, parse_response], 
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                characters=conv_key_characters,
                max_retry=5
            )
            
            # Normalize character name field
            try:
                for chara in response['key_characters']:
                    if 'name' not in chara and 'character' in chara:
                        chara['name'] = chara.pop('character')
            except:
                continue
                
            # Update conversation with enhanced content
            conversation['scenario'] = response['scenario']
            enhanced_motivations = {chara['name']: chara['motivation'] for chara in response['key_characters']}
            for chara in conversation['key_characters']:
                if 'name' not in chara and 'character' in chara:
                    chara['name'] = chara.pop('character')
                chara['motivation'] = enhanced_motivations[chara['name']]

    # Filter out plots with invalid character data
    plots = [
        p for p in plots 
        if all(['name' in c for c in p['key_characters']]) 
        and len(p['key_characters']) == len(set([c['name'] for c in p['key_characters']]))
    ]

    # Split plots into train/test sets
    split_index = int(len(plots) * 0.9)

    # Collect all character names
    character_names = set() 
    for plot in plots:
        # Add names from plot key characters
        for character in plot['key_characters']:
            character_names.add(character['name'])

        # Normalize conversation format
        if isinstance(plot['conversation'], dict) and all([k in plot['conversation'] for k in ['key_characters', 'dialogues']]):
            plot['conversation'] = [plot['conversation']]

        # Process each conversation
        for conversation in plot['conversation']:
            # Add names from conversation key characters
            for key_character in conversation['key_characters']:
                try:
                    character_names.add(key_character['name'])
                except:
                    logger.error(f"Error: {key_character}")
                    if 'character' in key_character:
                        key_character['name'] = key_character.pop('character')

            # Add names from dialogue utterances
            for utterance in conversation['dialogues']:
                try:
                    character_names.add(utterance['character'])
                except:
                    logger.error(f"Error: {utterance}")
                    if 'name' in utterance:
                        utterance['character'] = utterance.pop('name')
    
    # Sort character names
    character_names = sorted(character_names)

    # Generate prompt for standardizing character names
    prompt = """Given a list of character names, titles, or form of address, your task is to: i) generate a list of named characters with their official names (in {language}); ii) For each name in the given list, align it with the official character name if it refers to a named character, or denote it as "impersonal" otherwise.

===Output Format===
Please provide the output in the following JSON format:
{{
    "named_characters": [
        The list of named characters with their official names. Each character should appear only once. 
    ],
    "to_official_name": {{
        "The name in the list": "The official name of the character, or 'impersonal' if it does not refer to a named character."
    }}
}}
===Input===
{character_names}
"""

    prompt = prompt.replace('{character_names}', str(character_names))
    prompt = prompt.replace('{language}', language)

    # Helper function to validate name standardization response
    def parse_response(response, **kwargs):
        if 'named_characters' in response:
            return response
        else:
            return False

    # Get standardized names from LLM
    from utils import extract_json
    response = get_response_json(
        [extract_json, parse_response], 
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        max_retry=5
    )

    # Extract official names, falling back to candidate model if needed
    try:
        official_names = response['named_characters']
    except:
        response = get_response_json(
            [extract_json, parse_response],
            model=args.candidate_model,
            messages=[{"role": "user", "content": prompt}],
            max_retry=5
        )
        official_names = response['named_characters']
    
    # Normalize list values in name mapping
    for k, v in response['to_official_name'].items():
        if isinstance(v, list):
            response['to_official_name'][k] = v[0]

    # Add missing official names
    official_names += [
        n for n in set(response['to_official_name'].values()) - set(official_names)
        if n.lower() not in ['impersonal', 'environment']
    ]

    to_official_name = response['to_official_name']

    # Handle names missing from the mapping
    missing_names = []
    for name in character_names:
        if name not in to_official_name:
            print(f"Warning: {name} not included in to_official_name")
            missing_names.append(name)

    print(f"Missing names: {missing_names}")

    # Find closest matches for missing names
    for name in missing_names:
        closest_name = find_best_match_passage(official_names, name)
        print(f"Closest name: {closest_name}")
        if closest_name != -1:
            to_official_name[name] = official_names[closest_name]
        else:
            to_official_name[name] = "impersonal"

    # Initialize character datasets
    character_datasets = {
        character: {
            "plots": [],
            "conversations": [],
            "utterances": []
        } for character in official_names
    }

    # Populate character datasets
    for i_p, plot in enumerate(plots):
        # Process plot key characters
        for character in plot['key_characters']:
            if to_official_name[character['name']] != "impersonal":
                character['name'] = to_official_name[character['name']]
                character_datasets[character['name']]['plots'].append((i_p, character))

        # Process conversations
        for i_c, conversation in enumerate(plot['conversation']):
            # Process conversation key characters
            for character in conversation['key_characters']:
                if to_official_name[character['name']] != "impersonal":
                    character['name'] = to_official_name[character['name']]
                    character_datasets[character['name']]['conversations'].append((i_p, i_c, character))
            
            # Process utterances
            for i_u, utterance in enumerate(conversation['dialogues']):
                if to_official_name[utterance['character']] != "impersonal":
                    utterance['character'] = to_official_name[utterance['character']]
                    character_datasets[utterance['character']]['utterances'].append((i_p, i_c, i_u, utterance))

    # Generate character profiles prompt template
    prompt = """Please provide a concise, narrative-style character profile for {character_name} from "{book_title}". The profile should read like a cohesive introduction, weaving together the character's background, physical description, personality traits and core motivations, notable attributes, relationships, key experiences, major plot involvement and key decisions or actions, character arc or development throughout the story, and other important details. 
    
The profile should be written in a concise yet informative style, similar to what one might find in a comprehensive character guide, in {language}. Focus on the most crucial information that gives readers a clear understanding of the character's significance in the work. 

You will be provided with summaries and dialogues of some key plots in the book as reference. The profile should be based on either your existing knowledge of the character or the provided information, without fabricating or inferring any inaccurate or uncertain details. 

{character_data}

Now, please generate the character profile, starting with ===Profile===.
"""

    # Generate character profiles
    for character_name, character_data in character_datasets.items():
        # Get plots involving this character
        involved_plots = sorted(set(
            [p[0] for p in character_data['plots']] + 
            [c[0] for c in character_data['conversations']] + 
            [u[0] for u in character_data['utterances']]
        ))

        # Filter to training set plots only
        involved_plots = [i_p for i_p in involved_plots if i_p < split_index]
        
        # Collect plot information
        plot_infos = []
        for i_p in involved_plots:
            plot = plots[i_p]

            plot_info = {
                "plot": plot['summary'],
            }

            # Add character-specific information
            for key_character_info in plot['key_characters']:
                if key_character_info['name'] == character_name:
                    plot_info["character_experience"] = key_character_info

            # Add relevant conversations
            plot_info["conversation"] = []
            for conversation in plot['conversation']:
                if character_name in conversation['key_characters'] or character_name in [u['character'] for u in conversation['dialogues']]:
                    plot_info["conversation"].append(conversation)
            
            plot_infos.append(plot_info)

        

        character_prompt = prompt.replace("{character_name}", character_name).replace("{book_title}", book["title"]).replace("{character_data}", json.dumps(plot_infos, ensure_ascii=False, indent=2)).replace("{language}", language)

        print(character_prompt)

        # Get profile from LLM with retries
        nth_generation = 0
        while True:
            if nth_generation > 0:
                profile = get_response(
                    model=args.model,
                    messages=[{"role": "user", "content": character_prompt}],
                    nth_generation=nth_generation
                )
            else:
                profile = get_response(
                    model=args.model,
                    messages=[{"role": "user", "content": character_prompt}]
                )

            try:
                profile = profile.split("===Profile===", 1)[1].strip() 
                if profile.startswith('I apologize'): profile = ''
                character_datasets[character_name]['profile'] = profile
                break
            except:
                nth_generation += 1
                if nth_generation > 5:
                    character_datasets[character_name]['profile'] = ''
                    break
                continue

    # Update data format for readability
    for character_name, character_data in character_datasets.items():
        # Flatten plot data
        for i, plot in enumerate(character_data['plots']):
            character_data['plots'][i] = plot[-1]
            character_data['plots'][i]['i_p'] = plot[0]
        
        # Flatten conversation data
        for i, conversation in enumerate(character_data['conversations']):
            character_data['conversations'][i] = conversation[-1]
            character_data['conversations'][i]['i_p'] = conversation[0]
            character_data['conversations'][i]['i_c'] = conversation[1]
        
        # Flatten utterance data
        for i, utterance in enumerate(character_data['utterances']):
            character_data['utterances'][i] = utterance[-1]
            character_data['utterances'][i]['i_p'] = utterance[0]
            character_data['utterances'][i]['i_c'] = utterance[1]
            character_data['utterances'][i]['i_u'] = utterance[2]

    # Save final results
    results['character_datasets'] = character_datasets
    results['split_plot_index'] = split_index

    results.pop("chapter_beginnings")
    results.pop("fail_to_parse_responses")
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)



if __name__ == '__main__':

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Read input data
    with jsonlines.open(args.input, mode='r') as reader:
        books_data = list(reader)

    # Clean book titles
    for book in books_data:
        book['title'] = book['title'].replace('/', '-').replace(':', '_').replace('.', ' ')

    logger.info(f"Processing {len(books_data)} books")

    def process_book(book):
        try:
            extract(book)
            restore_from_cache(book)
            result = assemble(book)
            logger.info(f"Successfully processed book: {book.get('title', 'Unknown')}")
            return result
        except Exception as e:
            logger.error(f"Error processing book {book.get('title', 'Unknown')}: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    if args.num_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        
        logger.info(f"Starting parallel processing with {args.num_workers} workers")

        # Process books in parallel
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            processed_books = list(tqdm(
                executor.map(process_book, books_data),
                total=len(books_data),
                desc="Processing books"
            ))
    else:
        processed_books = []
        for book in tqdm(books_data):
            processed_book = process_book(book)
            processed_books.append(processed_book)














