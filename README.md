# VTGNet
[VTGNet: A Vision-based Trajectory Generation Network for Autonomous Vehicles in Urban Environments](https://sites.google.com/view/vtgnet/)

## Requirements
```
conda env create -f environment.yaml
```
This command will create a conda environment named `vtgnet`

## Test the model
```
conda activate vtgnet
python test.py
```

The result will be saved in folder `test_results/` as the following image:

<img src=test_results/vtgnet_result.jpg width="100%">

## VTG-Driving dataset
Download our dataset [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/pcaiaa_connect_ust_hk/EjWxfijz48lCvMWTsYLutegBQFmqNliFQsOZF60LbkwAXg?e=oVSt6h) and extract it into folder `VTG-Driving-Dataset/`

[Alternate download link](https://pan.baidu.com/s/1BPm-nXasXoJ_Ddxe42a6wQ) with code: 4xn8

```
VTG-Driving-Dataset
├─ clear_day/
├─ clear_sunset/
├─ foggy_day/
├─ rainy_day/
├─ rainy_sunset/
├─ dataset_left.csv
├─ dataset_right.csv
├─ dataset_straight.csv
```
This dataset is collected in [CARLA simulator](https://carla.org/). The driving information (location, speed, control actions, etc.) is recorded in *efo_info.csv* for each episode. The setups are listed as following:

* Dynamic traffic with pedestrians and other vehicles
* Collected in Town01 at desired speed of 40 km/h
* Five weather conditions
* 100 driving episodes for each weather
* 16.6 hours, 288.7 km
* **With behavoirs recovering from periodic off-center and off-orientation mistakes.**

The *dataset_{left, right, straight}.csv* files keeps the training information extracted from the dataset.

## Train the model on our dataset
```
python train.py --direction 2  --load_weights True --batch_size 15
```

--direction {0,1,2}  *0: keep straight; 1: turn right; 2: turn left*

--load_weights {True,False} *load pre-trained weights on Robotcar or not*

## Citation
```
@article{Cai2020VTGNetAV,
  title={VTGNet: A Vision-based Trajectory Generation Network for Autonomous Vehicles in Urban Environments},
  author={Peide Cai and Yuxiang Sun and H. Wang and M. Liu},
  journal={IEEE Transactions on Intelligent Vehicles},
  year={2020},
  doi={10.1109/TIV.2020.3033878}
}
```
