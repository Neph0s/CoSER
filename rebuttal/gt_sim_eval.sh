# 标准
python gca_evaluation/rbt_eval_yf.py --test_file data/test/rbt_test_set.json --judge_model gpt-4o --num_workers 5

# 用deepseek-r1评估，需要在gca_evaluation/utils.py中的get_response里添加调用v3的方式
python gca_evaluation/rbt_eval_yf.py --test_file data/test/rbt_test_set.json --judge_model deepseek-r1 --num_workers 5


