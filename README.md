##  Course Project: COMP 5331 Group 8 (22-fall)

#### Here we integrated the implementation of four proposed methods.

### For the two methods: EF-DTW and 10%-DTW, please check the branch "euclidean_dtw"

### For the two methods: Memory module and SFDW, please check the branch "memory"




#### Multivariate time series datasets:

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
