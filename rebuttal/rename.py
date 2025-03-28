
''' 
之前的数据名格式为: 
exp_name = f'1221-continue_from=0-200case-env_model=gpt-4o-nsp=llama3-1210-8epc'
simulation_path = f'./exp/results_old/{exp_name}-test_circumstance_id_{actor_model}.json'
evaluation_path = f'./exp/results_old/{exp_name}-test_circumstance_id_{actor_model}_eval.json'
'''

actor_models = ['groundtruth', 'llama3-1210-8epc', 'llama3p1-8b-0119', 'abab7-preview-chat', 'llama3p1-8b-instruct', 'gpt3.5', 'gpt-4o', 'claude35sonnet']
import json 

for actor_model in actor_models:
    if actor_model == 'groundtruth': 
        continue 

    exp_name = f'1221-continue_from=0-200case-env_model=gpt-4o-nsp=llama3-1210-8epc'
    simulation_path = f'./exp/results_old/{exp_name}-test_circumstance_ood_{actor_model}.json'
    evaluation_path = f'./exp/results_old/{exp_name}-test_circumstance_ood_{actor_model}_eval.json'

    # 读取simulation_path
    with open(simulation_path, 'r') as f:
        simulation_data = json.load(f)

    # 读取evaluation_path
    with open(evaluation_path, 'r') as f:
        evaluation_data = json.load(f)

    new_simulation_path = f'./exp/simulation/ood_set_{actor_model}.json'
    new_evaluation_path = f'./exp/evaluation/ood_set_{actor_model}.json'

    print(f'Successfully loaded data from {simulation_path} and {evaluation_path}')

    # 写入新的simulation_path
    with open(new_simulation_path, 'w') as f:
        json.dump(simulation_data, f, indent=2, ensure_ascii=False)

    # 写入新的evaluation_path
    with open(new_evaluation_path, 'w') as f:
        evaluation_data['scores'] = evaluation_data.pop('avg_scores')
        evaluation_data["cases"] = evaluation_data.pop('results_by_case')
        json.dump(evaluation_data, f, indent=2, ensure_ascii=False)