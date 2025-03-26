# 标准
# python gca_evaluation/rbt_eval.py --test_file data/test/id_set.json --judge_model gpt-4o --num_workers 5

# 用deepseek-v3评估，需要在gca_evaluation/utils.py中的get_response里添加调用v3的方式
python gca_evaluation/rbt_eval.py --test_file data/test/id_set.json --judge_model deepseek-v3 --num_workers 5

# 用deepseek-r1评估，需要在gca_evaluation/utils.py中的get_response里添加调用r1的方式
python gca_evaluation/rbt_eval.py --test_file data/test/id_set.json --judge_model deepseek-r1 --num_workers 5

# no length penalty 
# python gca_evaluation/rbt_eval.py --test_file data/test/id_set.json --judge_model gpt-4o --no_length_penalty --num_workers 5

# no rubric
python gca_evaluation/rbt_eval.py --test_file data/test/id_set.json --judge_model gpt-4o --no_rubric --num_workers 5

# no dimension separation
python gca_evaluation/rbt_eval.py --test_file data/test/id_set.json --judge_model gpt-4o --no_dim_sep --num_workers 5

# no reference
python gca_evaluation/rbt_eval.py --test_file data/test/id_set.json --judge_model gpt-4o --no_reference --num_workers 5