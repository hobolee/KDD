
Datasets -, SOLAR and ENERGY

## Experiments on the proposed method
```
python train_multi_step.py --data ./data/{0} --model_name {1} --device cuda:0 --expid {2} --epochs 0 --batch_size 64 --runs 10 --random_node_idx_split_runs 100 --lower_limit_random_node_selections 15 --upper_limit_random_node_selections 15 --borrow_from_train_data true --num_neighbors_borrow 5 --dist_exp_value 0.5 --neighbor_temp 0.1 --use_ewp True --memory_separate False --memory_integrate True --SFDW True
```
* --memory_separate True if you want to enable it.
* --memory_integrate True if you want to enable it.
* --SFDW True if you want to enable it.

## Requirements
The model is implemented using Python3 with dependencies specified in requirements.txt