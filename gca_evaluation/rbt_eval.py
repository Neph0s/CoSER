import json 
from utils import get_response
import argparse
from tqdm import tqdm
from utils import setup_logger
from agent import Agent
import random
import os
from utils import get_environment_prompt, get_nsp_prompt, get_character_prompt
from utils import get_response_json, extract_json
from utils import remove_inner_thoughts, calculate_bleu_rouge
random.seed(42)

logger = None

# Set up command line argument parser
parser = argparse.ArgumentParser(
    description='Evaluate role-playing language models via given-circumstance acting (GCA)'
)

# Input/output paths
parser.add_argument(
    '--test_file',
    type=str,
    default='data/test/test_set.json',
    help='Path to the test dataset'
)
parser.add_argument(
    '--book_data',
    type=str,
    default='data/final',
    help='Path to the folder containing complete curated data of each book, used when retrieval augmentation is enabled.'
)

# Model configuration
parser.add_argument(
    '--actor_model',
    type=str,
    default='gpt-4o',
    help='Name of the model to use for role-playing'
)
parser.add_argument(
    '--judge_model',
    type=str,
    default='gpt-4o',
    help='Name of the model to use for LLM judging'
)
parser.add_argument(
    '--env_model',
    type=str,
    default='gpt-4o',
    help='Name of the model to use for environment response'
)
parser.add_argument(
    '--nsp_model',
    type=str,
    default='gpt-4o-mini',
    help='Name of the model to use for next-speaker prediction, default to gpt-4o-mini, but recommend Coser-70B or self-deployed models for better cost-efficiency.'
)

# Runtime settings
parser.add_argument(
    '--continue_from',
    type=int,
    default=0,
    help='Start GCA from the i-th round. The previous rounds will use the ground truth conversations.'
)
parser.add_argument(
    '--wo_thought',
    default=False,
    action='store_true',
    help='Disable inner thoughts in generation'
)
parser.add_argument(
    '--retrieval',
    type=str,
    default=None,
    choices=[None, 'raw_text', 'expr1', 'expr3', 'conv1', 'expr3_conv1', 'expr10_conv1'],
    help='Target for retrieval'
)
parser.add_argument(
    '--regenerate',
    action='store_true',
    help='Regenerate the simulation results'
)
parser.add_argument(
    '--reevaluate',
    action='store_true',
    help='Reevaluate the simulation results'
)
parser.add_argument(
    '--nth_exp',
    type=int,
    default=0,
    help='Experiment ID. Results will be reused for same ID. Set to -1 to run 3 experiments.'
)
parser.add_argument(
    '--num_workers',
    type=int,
    default=1,
    help='Number of parallel workers (default: 1)'
)

parser.add_argument(
    '--no_length_penalty',
    default=False,
    action='store_true',
)

parser.add_argument(
    '--no_rubric',
    default=False,
    action='store_true',
)

parser.add_argument(
    '--no_reference',
    default=False,
    action='store_true',
)

parser.add_argument(
    '--no_dim_sep',
    default=False,
    action='store_true',
)

# Parse arguments
args = parser.parse_args()

ENVIRONMENT = 'Environment'
NSP = "NSP"
special_characters = [NSP, ENVIRONMENT]

def gca_simulation(test_file, actor_model, env_model, nsp_model, retrieval, nth_exp=0):
    """
    Conducts Given-Circumstance Acting (GCA) simulation where LLM agents role-play characters in specific scenarios.
    The simulation involves multiple agents:
    - Character agents (using actor_model) that role-play the characters
    - Environment agent (using env_model) that takes a special role of "Environment" and provides environmental feedback
    - Next-Speaker Predictor (using nsp_model) that determines the speaking agent in each round

    Each character agent is initialized with relevant character data. 
    The agents then engage in multi-turn dialogue, with the NSP model directing speaker transitions.

    Args:
        test_file (str): Path to JSON file containing test cases.
        actor_model (str): Model name for character role-playing agents
        env_model (str): Model name for environment agent
        nsp_model (str): Model name for next speaker prediction
        retrieval (str, optional): Type of retrieval data to enhance role-playing. Defaults to None (no retrieval).
        nth_exp (int, optional): Experiment ID. 

    Returns:
        list: Simulation results for each scenario.
    """

    # Set up caching file for model outputs
    from utils import set_cache_path
    cache_path = f'.cache/{actor_model}.pkl'
    if nth_exp > 0:
        cache_path = f'{cache_path}-repeat={nth_exp}'
    set_cache_path(cache_path)
    
    # Load test set
    test_dataset = json.load(open(test_file, 'r'))
    results = []

    # Configure output path based on model and retrieval settings
    actor_setting = f'{actor_model}{"_rag=" + retrieval if retrieval else ""}'
    simulation_path = f'exp/simulation/{test_file.split("/")[-1].replace(".json", "")}_{actor_setting}.json'

    logger.info(f'Conducting GCA Simulation for {actor_setting} on {test_file}\n\nThe results will be saved to {simulation_path}')
    
    # Return cached results if available 
    if os.path.exists(simulation_path) and not args.regenerate:
        return json.load(open(simulation_path, 'r'))

    # Traverse each test sample in the test dataset
    for circumstance in test_dataset:
        # collect scenario metadata and context
        book_title = circumstance['book']
        plot = circumstance['plot']
        i_p = plot['i_p'] 
        conversation = circumstance
        i_c = conversation['i_c']
        character_profiles = circumstance['character_profiles']

        logger.info(f'==========Book {book_title}==========')

        # Load additional book data if retrieval is enabled
        if retrieval:
            book_database = json.load(open(f'{args.book_data}/{book_title}.json', 'r'))

        
        # Identify the character lists
        plot_characters = [ c['name'] for c in plot['key_characters']] 
        speaking_characters_w_env = conversation['speaking_characters_w_env']
        if ENVIRONMENT not in speaking_characters_w_env:
            speaking_characters_w_env.append(ENVIRONMENT)
        major_characters = conversation['major_characters']

        character_agents = {}
        involved_character_profiles = {}

        # Build enhanced character profiles combining scenario and plot information
        for character in speaking_characters_w_env:    
            if character == ENVIRONMENT:
                continue
            
            character_profile = character_profiles.get(character, '')
            if character in plot_characters:
                character_info = [c for c in plot['key_characters'] if c.get('name', '') == character][0]
                if 'description' in character_info:
                    character_profile = character_info.get('description', '').strip('\n') + '\n\n' + character_profile.strip('\n')
                    
            character_profile = character_profile.strip(' \n')
            if character_profile != '':
                involved_character_profiles[character] = character_profile

        if actor_model == 'groundtruth': 
            agent_conversations = []
            for m in circumstance['dialogues']:
                agent_conversations.append({"role": m['character'], "content": m['character'] + ': ' + m['message'].strip(' ')})
        else:
            # Create agents for all roles (characters + NSP)
            for character in speaking_characters_w_env + [NSP]:    
                # Configure agent based on role type
                if character == NSP:
                    # Next Speaker Predictor agent
                    system_prompt = get_nsp_prompt(speaking_characters_w_env, conversation['scenario'])
                    character_database = None
                elif character == ENVIRONMENT:
                    # Environment description agent
                    system_prompt = get_environment_prompt(major_characters, conversation['scenario'])
                    character_database = None
                else:
                    # Character role-playing agent
                    if retrieval and character in book_database['character_datasets']:
                        # Set up retrieval database for character context
                        character_database = book_database['character_datasets'][character]
                        involved_plots = [_['i_p'] for _ in character_database['plots']] + \
                                    [_['i_p'] for _ in character_database['conversations']] + \
                                    [_['i_p'] for _ in character_database['utterances']]
                        involved_plots = sorted(set(involved_plots))
                        character_database['detailed_plots'] = [ book_database['plots'][i] for i in involved_plots ] 
                    else:
                        character_database = None

                    # Build character context from profile and plot
                    character_profile = involved_character_profiles.get(character, '')
                    if character in plot_characters:
                        character_info = [c for c in plot['key_characters'] if c.get('name', '') == character][0]
                    character_profile = character_profile.strip(' \n')

                    # Get character motivation if specified
                    find_motivation = [ c.get('motivation', '') for c in conversation['key_characters'] if c.get('name', '') == character]
                    motivation = find_motivation[0] if find_motivation else ''

                    # Configure prompt based on model type
                    add_output_example = False if 'coser' in actor_model.lower() else True
                    system_prompt = get_character_prompt(
                        book_title, character, character_profile, plot["summary"],
                        conversation["scenario"], motivation, thoughtless=args.wo_thought,
                        other_character_profiles=involved_character_profiles,
                        exclude_plot_summary=True, fixed_template=True,
                        add_output_example=add_output_example, add_rag=retrieval
                    )

                # Select appropriate model for the agent
                if character not in special_characters:
                    character_model = actor_model  # Character role-playing
                elif character == ENVIRONMENT:
                    character_model = env_model    # Environment description
                elif character == NSP:
                    character_model = nsp_model    # Next speaker prediction
                else:
                    raise ValueError(f'Invalid character: {character}')

                # Initialize the agent with its configuration
                character_agent = Agent(
                    character_model, character, character_database,
                    system_prompt=system_prompt,
                    retrieval_target=retrieval if (retrieval and character not in special_characters) else None
                )
                character_agent.update('user', "===Conversation Start===\n\n")
                character_agents[character] = character_agent

            # Begin conversation simulation
            max_rounds = 20
            agent_conversations = []
            current_speaker = speaking_characters_w_env[0]  # Start with first character
            
            # Main conversation loop
            for i_round in range(max_rounds):
                if current_speaker == "<END CHAT>":
                    break

                logger.info(f'===Round {i_round}===\n')
                
                # Generate responses for current speaker and get next speaker prediction
                for actor in [current_speaker, "NSP"]:
                    current_agent = character_agents[actor]
                    from utils import add_speaker_name
                    
                    # Use ground truth for early rounds if specified
                    if args.continue_from > i_round:
                        if actor == current_speaker:
                            response = conversation['dialogues'][i_round]['message']
                        else:  # NSP
                            response = conversation['dialogues'][i_round+1]['character'] if i_round < len(conversation['dialogues']) - 1 else '<END CHAT>'
                    else:
                        response = current_agent.chat()

                    if actor == "NSP":
                        # Process next speaker prediction
                        next_actor = response.split(':')[0].strip() if ':' in response else response

                        # Validate and set next speaker
                        if next_actor == "<END CHAT>" and i_round >= 5:
                            current_speaker = "<END CHAT>"
                        elif next_actor in speaking_characters_w_env and next_actor != current_speaker:
                            current_speaker = next_actor
                        else:
                            # Fallback to random selection if prediction is invalid
                            candidates = set(major_characters + [ENVIRONMENT]) - {current_speaker}
                            if not candidates:
                                candidates = set(speaking_characters_w_env) - {current_speaker}
                            candidates = list(candidates)
                            current_speaker = random.choice(candidates)
                        
                        logger.info(f"Next speaker: {current_speaker} (Raw response: {response})")
                        agent_conversations.append({"role": actor, "content": next_actor})
                        current_agent.update('assistant', next_actor)
                    
                    else:
                        # Process character/environment response
                        response = add_speaker_name(response, actor)
                        logger.info(f"{env_model if actor == ENVIRONMENT else actor_model}: {response}\n")
                        agent_conversations.append({"role": actor, "content": response})

                        # Update conversation history for all agents
                        for other_actor, other_agent in character_agents.items():
                            if other_actor == actor:
                                other_agent.update('assistant', response)
                            else:
                                other_agent.update('user', remove_inner_thoughts(response))

        # Store simulation results
        results.append({
            'book_title': book_title,
            'i_p': i_p,
            'i_c': i_c,
            'circumstance': circumstance,
            'simulation': agent_conversations,
            'involved_character_profiles': involved_character_profiles
        })


    # Save simulation results
    os.makedirs(os.path.dirname(simulation_path), exist_ok=True)
    with open(simulation_path, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results

def gca_judging(test_file, actor_model, retrieval, judge_model, nth_exp=0):
    """
    Evaluates the quality of Given-Circumstance Acting (GCA) simulation results using multiple metrics.
    
    This function loads simulation results and evaluates them against reference dialogues using both automated metrics (BLEU, ROUGE-L) and LLM-based judgments across four dimensions:
    - Storyline Consistency: Measures alignment between the simulated conversation and original dialogue 
    - Anthropomorphism: Evaluates whether RPLAs behave in a human-like manner
    - Character Fidelity: Assesses whether RPLAs faithfully portray their characters
    - Storyline Quality: Evaluates whether the simulated conversation develops naturally

    Args:
        test_file (str): Path to JSON file containing test cases
        actor_model (str): Model name for character role-playing agents
        retrieval (str, optional): Type of retrieval data to enhance role-playing. Defaults to None (no retrieval).
        judge_model (str): Model name for LLM Judges.
        nth_exp (int, optional): Experiment ID.

    Returns:
        tuple: (avg_scores, cases)
            - avg_scores (dict): Average scores across all evaluation metrics
            - cases (dict): Detailed evaluation results for each test case
    """
    from utils import set_cache_path

    # Set up caching file for model outputs
    cache_path = f'.cache/{actor_model}.pkl'
    if nth_exp > 0:
        cache_path = f'{cache_path}-repeat={nth_exp}'
    set_cache_path(cache_path)
    
    # Configure paths based on model and retrieval settings
    actor_setting = f'{actor_model}{"_rag=" + retrieval if retrieval else ""}'
    simulation_path = f'exp/simulation/{test_file.split("/")[-1].replace(".json", "")}_{actor_setting}.json'
    evaluation_path = simulation_path.replace('/simulation/', '/evaluation/').replace('.json', f'{exp_name.replace("eval", "")}.json')

    logger.info(f'Evaluating GCA Simulation for {actor_setting} on {test_file}\n\nThe results will be saved to {evaluation_path}')
    
    # Return cached evaluation results if available
    if os.path.exists(evaluation_path) and not (args.regenerate or args.reevaluate):
        res = json.load(open(evaluation_path, 'r'))
        return res['scores'], res['cases']

    if args.no_length_penalty:
        orig_res = json.load(open(evaluation_path.replace('-no_length_penalty', ''), 'r'))
    
    # Load the simulation results
    simulation_results = json.load(open(simulation_path, 'r'))

    # Define evaluation dimensions
    dimensions = ['Storyline Consistency', 'Anthropomorphism', 'Character Fidelity', 'Storyline Quality']

    if args.no_reference:
        dimensions = ['Anthropomorphism', 'Character Fidelity', 'Storyline Quality']

    scores = { d: [] for d in dimensions + ['bleu', 'rouge_l'] }
    cases = {}

    # Evaluate each simulation result
    for result in simulation_results:
        book_title, i_p, i_c, circumstance, simulation = result['book_title'], result['i_p'], result['i_c'], result['circumstance'], result['simulation'] 
        
        # Verify indices match
        assert i_p == circumstance['plot']['i_p']
        assert i_c == circumstance['i_c']

        logger.info(f'Book {book_title}')

        # Filter out NSP messages and clean up simulation/reference for comparison
        simulation = result['simulation']
        simulation = [m for m in simulation if m['role'] != NSP]
        reference = circumstance['dialogues']

        # Remove inner thoughts for fair comparison
        simulation = [ m if m['role'] == ENVIRONMENT else 
            {**m, 'content': remove_inner_thoughts(m['content'])} 
            for m in simulation  ]

        reference = [ m if m['character'] == ENVIRONMENT else 
            {**m, 'message': remove_inner_thoughts(m['message'])} 
            for m in reference  ]
        
        # Convert to readable string format for evaluation
        simulation_str = '\n\n'.join([m['content'].strip('\n') for m in simulation])
        reference_str = '\n\n'.join([f"{m['character']}: {m['message']}".strip('\n') for m in reference])
            
        logger.info(f'===Simulation of {actor_setting}===\n\n**************\n{simulation_str}\n\n**************\n\n===Reference===\n\n**************\n{reference_str}\n\n**************\n\n')

        # Prepare context information for evaluation
        scenario_str =  circumstance['scenario']
        character_profile_str = '\n\n'.join([f"### {character}\n\n{profile.strip('')}" for character, profile in result['involved_character_profiles'].items()])
        major_characters = circumstance['major_characters']

        # Add special instructions for partial evaluation if needed
        additional_instructions = ''
        if args.continue_from > 0:
            additional_instructions = f'Please note that the first {args.continue_from} messages in the simulated conversation are the same as the reference. Focus your evaluation only on the content after these messages.'

        # Helper function to validate evaluation response format
        def parse_response(response, **kwargs):
            try:
                assert isinstance(response, dict)
                for k, v in response.items():
                    assert k in dimensions
                    assert 'flaws' in v

                    for f in v['flaws']:
                        if f.get('severity', None) is None:
                            f['severity'] = 1

                return response
            except:
                return False

        logger.info(f'{book_title}-{i_p}-{i_c}-{scenario_str}')

        # Count non-environment messages for score adjustment
        actor_rounds = len([m for m in simulation if m['role'] != ENVIRONMENT])
        eval_result = {}

        # Evaluate each dimension using LLM
        for dimension in dimensions:
            from prompts import critic_prompts

            if args.no_rubric:
                critic_prompt = critic_prompts['self-play-deduct-template'].replace('{book}', book_title).replace('{plot_summary}', circumstance['plot']['summary']).replace('{scenario}', scenario_str).replace('{character_profiles}', character_profile_str).replace('{original_conversation}', reference_str).replace('{major_characters}', ', '.join(major_characters)).replace('{additional_instructions}', additional_instructions).replace('{dimension_name}', dimension).replace('{dimension_brief}', critic_prompts['dimension_details'][dimension]['dimension_brief']).replace('{dimension_criteria}', critic_prompts['dimension_details'][dimension]['dimension_brief'])
            elif args.no_dim_sep:
                critic_prompt = critic_prompts['self-play-deduct-template'].replace('{book}', book_title).replace('{plot_summary}', circumstance['plot']['summary']).replace('{scenario}', scenario_str).replace('{character_profiles}', character_profile_str).replace('{original_conversation}', reference_str).replace('{major_characters}', ', '.join(major_characters)).replace('{additional_instructions}', additional_instructions).replace('{dimension_name}', ', '.join(dimensions), 1).replace(', i.e., {dimension_brief}', '').replace('{dimension_criteria}', '\n\n'.join([ d + ':\n\n' +critic_prompts['dimension_details'][d]['dimension_criteria'] for d in dimensions])).replace('"{dimension_name}": {', '"Storyline Consistency": { // Make sure to output in four dimensions: Storyline Consistency, Anthropomorphism, Character Fidelity, and Storyline Quality. ')
            elif args.no_reference:
                critic_prompt = critic_prompts['self-play-deduct-template'].replace('* The original conversation from {book} in the same scenario as a reference.', '').replace('{book}', book_title).replace('{plot_summary}', circumstance['plot']['summary']).replace('{scenario}', scenario_str).replace('{character_profiles}', character_profile_str).replace('{original_conversation}', '').replace('## Original Conversation', '').replace('{major_characters}', ', '.join(major_characters)).replace('{additional_instructions}', additional_instructions).replace('{dimension_name}', dimension).replace('{dimension_brief}', critic_prompts['dimension_details'][dimension]['dimension_brief']).replace('{dimension_criteria}', critic_prompts['dimension_details'][dimension]['dimension_criteria'])
            else:
                critic_prompt = critic_prompts['self-play-deduct-template'].replace('{book}', book_title).replace('{plot_summary}', circumstance['plot']['summary']).replace('{scenario}', scenario_str).replace('{character_profiles}', character_profile_str).replace('{original_conversation}', reference_str).replace('{major_characters}', ', '.join(major_characters)).replace('{additional_instructions}', additional_instructions).replace('{dimension_name}', dimension).replace('{dimension_brief}', critic_prompts['dimension_details'][dimension]['dimension_brief']).replace('{dimension_criteria}', critic_prompts['dimension_details'][dimension]['dimension_criteria'])

            if args.no_length_penalty:
                res = {dimension: orig_res['cases'][actor_model][f'{book_title}-{i_p}-{i_c}']["critique"][dimension]}
            elif not (args.no_dim_sep and dimension != dimensions[0]):
                res = get_response_json([extract_json, parse_response], model=judge_model, messages=[{"role": "system", "content": critic_prompt}, {"role": "user", "content": simulation_str}])
            
            eval_result.update({dimension: res[dimension]})
            
            logger.info(json.dumps(res, ensure_ascii=False, indent=2)) 
            
            if args.no_length_penalty:
                # Calculate dimension score with length penalty
                res[dimension]['score'] = max(0, min(100 - (sum([f['severity'] for f in res[dimension]['flaws'] if isinstance(f['severity'], int)])) * 5, 100) )
            else:
                # Calculate dimension score with length penalty
                res[dimension]['score'] = max(0, min(100 - (sum([f['severity'] for f in res[dimension]['flaws'] if isinstance(f['severity'], int)]) - 0.3 * actor_rounds) * 5, 100) )
            

        # Calculate automated metrics
        bleu, rouge_l = calculate_bleu_rouge(reference[args.continue_from:], simulation[args.continue_from:])
        eval_result['bleu'] = bleu
        eval_result['rouge_l'] = rouge_l

        # Store evaluation results
        cases[f'{book_title}-{i_p}-{i_c}'] = {
            'simulation': simulation,
            'simulation_str': simulation_str,
            'score': sum([eval_result[dimension]['score'] for dimension in dimensions]) / len(dimensions),
            'critique': eval_result,
        }

        # Accumulate scores
        for dimension in dimensions:
            scores[dimension].append(eval_result[dimension]['score'])
        scores['bleu'].append(bleu)
        scores['rouge_l'].append(rouge_l)

    # Calculate average scores across all dimensions
    avg_scores = {dimension: sum(scores[dimension]) / max(1, len(scores[dimension])) for dimension in dimensions}
    avg_scores['avg'] = sum(avg_scores.values()) / len(avg_scores)
    avg_scores.update({metric: sum(scores[metric]) / max(1, len(scores[metric])) for metric in ['bleu', 'rouge_l']})

    logger.info(f'{actor_setting}: Average score of {len(simulation_results)} samples: \n{avg_scores["avg"]} {avg_scores} on {test_file}')

    # Save evaluation results
    os.makedirs(os.path.dirname(evaluation_path), exist_ok=True)
    with open(evaluation_path, 'w') as f:
        json.dump({'scores': avg_scores, 'cases': cases}, f, ensure_ascii=False, indent=2)

    return avg_scores, cases



if __name__ == "__main__":

    if args.nth_exp >= 0:
        nth_exps = [args.nth_exp]
    else:
        repeat_times = 3
        nth_exps = range(repeat_times)

    # Run experiments for each repeat
    for nth_exp in nth_exps:
        # Configure experiment name and logging
        exp_name = 'eval' 
        if args.continue_from > 0: exp_name += f'-continue_from={args.continue_from}' 
        if args.no_length_penalty: exp_name += '-no_length_penalty'
        if args.no_rubric: exp_name += '-no_rubric'
        if args.no_dim_sep: exp_name += '-no_dim_sep'
        if args.no_reference: exp_name += '-no_reference'
        if nth_exp > 0: exp_name += f'-repeat={nth_exp}'
        if args.judge_model != 'gpt-4o': exp_name += f'-judge={args.judge_model}'
        
        logger = setup_logger(__name__, f'{__file__.split(".")[0]}-{exp_name}.log')

        # Initialize result storage
        all_cases = {} 
        all_scores = {} 

        from concurrent.futures import ProcessPoolExecutor
        import functools

        def generate(exp_args):
            """Run simulation for given experiment args"""
            actor_model, args, nth_exp = exp_args
        
            results = gca_simulation(
                args.test_file,
                actor_model, 
                args.env_model,
                args.nsp_model,
                args.retrieval,
                nth_exp
            )

            return results

        def evaluate(exp_args):
            """Run evaluation for given experiment args"""
            actor_model, args, nth_exp = exp_args

            scores, cases = gca_judging(
                args.test_file,
                actor_model,
                args.retrieval,
                args.judge_model,
                nth_exp
            )

            return scores, cases
        
        # List of actor models to evaluate
        actor_models = ['llama3-1210-8epc', 'llama3p1-8b-0119', 'abab7-preview-chat', 'llama3p1-8b-instruct', 'gpt3.5', 'gpt-4o', 'claude35sonnet'] # you can modify the list to expand to multiple models
        if args.no_reference:
            actor_models += ['groundtruth']


        # Create experiment args for each actor model
        exp_args = [(actor_model, args, nth_exp) for actor_model in actor_models]

        # Parallel execution path when multiple workers available
        if args.num_workers > 1 and len(exp_args) > 1:
            # First run all generate tasks simultaneously
            generate_futures = []
            with ProcessPoolExecutor(max_workers=args.num_workers) as generate_executor:
                for exp_arg in exp_args:
                    future = generate_executor.submit(generate, exp_arg)
                    generate_futures.append((future, exp_arg))
            
            # As generate tasks complete, run evaluate tasks in parallel
            with ProcessPoolExecutor(max_workers=args.num_workers) as evaluate_executor:
                evaluate_futures = []
                
                # Process completed generate tasks and submit evaluates
                for generate_future, exp_arg in generate_futures:
                    generate_future.result()  # Wait for generate to complete
                    future = evaluate_executor.submit(evaluate, exp_arg)
                    evaluate_futures.append((future, exp_arg))
                
                # Process evaluate results as they complete
                for evaluate_future, exp_arg in evaluate_futures:
                    scores, cases = evaluate_future.result()

                    actor_model = exp_arg[0]
                    # Create identifier for this model run
                    actor_setting = f'{actor_model}{"_rag=" + args.retrieval if args.retrieval else ""}'

                    all_scores[actor_setting] = scores
                    all_cases[actor_setting] = cases

        # Sequential execution path
        else:
            for exp_arg in exp_args:
                generate(exp_arg)
                scores, cases = evaluate(exp_arg)

                actor_model = exp_arg[0]
                # Create identifier for this model run
                actor_setting = f'{actor_model}{"_rag=" + args.retrieval if args.retrieval else ""}'

                all_scores[actor_setting] = scores
                all_cases[actor_setting] = cases
                
        # Log final results
        logger.info(f'Evaluation results:\n{json.dumps(all_scores, ensure_ascii=False, indent=2)}')