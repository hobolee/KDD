##  Course Project: COMP 5331 Group 8 (22-fall)

#### Here we integrated the implementation of four proposed methods.

### For the two methods: EF-DTW and 10%-DTW, please check the branch "euclidean_dtw"

### For the two methods: Memory module and SFDW, please check the branch "memory"


### Running the model

Datasets - SOLAR, ENERGY. This code provides a running example with all components on [MTGNN](https://github.com/nnzhan/MTGNN) model (we acknowledge the authors of the work).

#### Standard Training
```
python train_multi_step.py --data ./data/{0} --model_name {1} --device cuda:0 --expid {2} --epochs 100 --batch_size 64 --runs 10 --random_node_idx_split_runs 100 --lower_limit_random_node_selections 15 --upper_limit_random_node_selections 15 --step_size1 {3} --mask_remaining {4}
```
Here, <br />
{0} - refers to the dataset directory: ./data/{ENERGY/SOLAR} <br />
{1} - refers to the model name <br />
{2} - refers to the manually assigned "ID" of the experiment  <br />
{3} - step_size1 is 2500 for SOLAR. <br />
{4} - inference post training in the partial setting, set to true or false. Note - mask_remaining is the alias for "Partial" setting in the original paper
* random_node_idx_split_runs - the number of randomly sampled subsets per trained model run
* lower_limit_random_node_selections and upper_limit_random_node_selections - the percentage of variables in the subset **S**.


#### Training with predefined subset S, the S apriori setting
```
python train_multi_step.py --data ./data/{0} --model_name {1} --device cuda:0 --expid {2} --epochs 100 --batch_size 64 --runs 50 --predefined_S --random_node_idx_split_runs 1 --lower_limit_random_node_selections 100 --upper_limit_random_node_selections 100 --step_size1 {3}
```


#### Training the model with Identity matrix as Adjacency
```
python train_multi_step.py --data ./data/{0} --model_name {1} --device cuda:0 --expid {2} --epochs 100 --batch_size 64 --runs 10 --adj_identity_train_test --random_node_idx_split_runs 100 --lower_limit_random_node_selections 100 --upper_limit_random_node_selections 100 --step_size1 {3}
```

### Multivariate time series datasets:

Download Solar dataset from [https://github.com/laiguokun/multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data). Uncompress them and move them to the data folder.

Download Energy dataset from

```

# Create data directories
mkdir -p data/{SOLAR,ENERGY}

# for any dataset, run the following command
python generate_training_data.py --ds_name {0} --output_dir data/{1} --dataset_filename data/{2}
```
Here <br />
{0} is for the dataset: solar, energy <br />
{1} is the directory where to save the train, valid, test splits. These are created from the first command <br />
{2} the raw data filename (the downloaded file), such as - solar.txt.
