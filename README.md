##  Course Project: COMP 5331 Group 8 (22-fall)

#### Here we integrated the implementation of four proposed methods.

### For the EF-DTW and 10%-DTW, please check the branch "euclidean_dtw"

### For the Memory module and SFDW, please check the branch "memory"

#### Multivariate time series datasets:

Download Solar and Traffic datasets from [https://github.com/laiguokun/multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data). Uncompress them and move them to the data folder.

Download the METR-LA dataset from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) provided by [Li et al.](https://github.com/liyaguang/DCRNN.git). Move them into the data folder. (Optinally - download the adjacency matrix for META-LA from [here](https://github.com/nnzhan/MTGNN/blob/master/data/sensor_graph/adj_mx.pkl) and put it as ./data/sensor_graph/adj_mx.pkl , as shown below):
```
wget https://github.com/nnzhan/MTGNN/blob/master/data/sensor_graph/adj_mx.pkl
mkdir data/sensor_graph
mv adj_mx.pkl data/sensor_graph/
```

Download the ECG5000 dataset from [time series classification](http://www.timeseriesclassification.com/description.php?Dataset=ECG5000).

```

# Create data directories
mkdir -p data/{METR-LA,SOLAR,TRAFFIC,ECG}

# for any dataset, run the following command
python generate_training_data.py --ds_name {0} --output_dir data/{1} --dataset_filename data/{2}
```
Here <br />
{0} is for the dataset: metr-la, solar, traffic, ECG <br />
{1} is the directory where to save the train, valid, test splits. These are created from the first command <br />
{2} the raw data filename (the downloaded file), such as - ECG_data.csv, metr-la.hd5, solar.txt, traffic.txt
