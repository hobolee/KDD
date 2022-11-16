# EF-DTW and 10%-DTW

## Settings
### Partial setting inference
```
python test.py --data ./data/ENERGY --model_name tri1 --expid 1 --runs 1 --random_node_idx_split_runs 100 --lower_limit_random_node_selections 15 --upper_limit_random_node_selections 15 --mask_remaining True
```
* Note that epochs are set to 0 and mask_remaining (alias of "Partial" setting in the paper) to True


### Oracle setting inference
```
python test.py --data ./data/ENERGY \
--model_name tri1 --expid 1 --epochs 0  --runs 1  --lower_limit_random_node_selections 100 \
 --upper_limit_random_node_selections 100 --do_full_set_oracle True --full_set_oracle_lower_limit 15 --full_set_oracle_upper_limit 15
```

## Three different methods
### Author's Wrapper Technique
```
python test.py --data ./data/ENERGY \
 --model_name tri1  --expid 1  --runs 1 --random_node_idx_split_runs 10 --borrow_from_train_data True --num_neighbors_borrow 5 --dist_exp_value 0.5 --neighbor_temp 0.1 --use_ewp True --obtain_relevant_data_methods 1

```
### Our Using 10% data with DTW method
```
python test.py --data ./data/ENERGY \
 --model_name tri1  --expid 1  --runs 1 --random_node_idx_split_runs 10 --borrow_from_train_data True --num_neighbors_borrow 5 --dist_exp_value 0.5 --neighbor_temp 0.1 --use_ewp True --obtain_relevant_data_methods 2
```

### Our Euclidean distance + DTW method
```
python test.py --data ./data/ENERGY \
 --model_name tri1  --expid 1  --runs 1 --random_node_idx_split_runs 10 --borrow_from_train_data True --num_neighbors_borrow 5 --dist_exp_value 0.5 --neighbor_temp 0.1 --use_ewp True --obtain_relevant_data_methods 3
```

## Requirements
The model is implemented using Python3 with dependencies specified in requirements.txt


## Data Preparation


### Multivariate time series datasets

Please refers haobo's branch(memory) or the kdd paper's repo to get the pre-processed datasets.
