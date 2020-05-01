# DL Final Project Submission
Code for the Deep Learning project - habitat challenge PointNav CVPR 2020 

Github Repo: https://github.com/akshay-krishnan/habitat-submission.git 


## DEPENDENCY

- [Habitat-Sim](https://github.com/facebookresearch/habitat-sim)
- [Habitat-API](https://github.com/facebookresearch/habitat-api)

## Data Setup
- Follow Habitat instruction to download gibson dataset
- Note the `data/` location
- Specify location in train config files (`*.yaml`) used in our code base.

# Main Implementation

> Modified [Neural-SLAM](https://github.com/devendrachaplot/Neural-SLAM)'s architecture for [PointGoal Habitat Challenge 2020](https://github.com/facebookresearch/habitat-challenge/tree/8ef39499bdaa4b73aa8968fac7bb068c296b79e0)

## Motivation
- Navigate without GPS/Compass sensor
- Make agent robust to actuation, RGB, and Depth sensor noises

## Contribution - Learn pose estimator explicitely

We do this by first learning an accuracte pose estimator using classical shortest path follower algorithm (TODO name)

### To train pose estimator
Code for this can be found here: [Neural-SLAM/pose_estimation_optimal](Neural-SLAM/pose_estimation_optimal)

```
cd Neural-SLAM
python pose_estimation_optimal.py \
    --split train_small \
    --train_slam 1 \
    --load_slam tmp/models/FinalTrainExp3/model_best.slam \
    --num_processes 1 \
    --num_processes_per_gpu 14 \
    --task_config "../configs/train_pose_estimation.local.rgbd.yaml" \
    --exp_name FinalTrainExp3 \
    --print_images 1 \
    --print_frequency 25 \
    --max_episode_length 1000 \
    --log_interval 50 \
    --eval 0 \
    --vis_type 2
```

### To evaluate pose estimator
Code for this can be found here: [Neural-SLAM/pose_estimation_optimal_eval.py](Neural-SLAM/pose_estimation_optimal_eval.py)

```
cd Neural-SLAM;
python pose_estimation_optimal.py \
    --split train_small \
    --train_slam 0 \
    --load_slam tmp/models/FinalTrainExp3/model_best.slam \
    --num_processes 1 \
    --num_processes_per_gpu 14 \
    --task_config "../configs/train_pose_estimation.local.rgbd.yaml" \
    --exp_name FinalTrainExp3 \
    --print_images 1 \
    --print_frequency 25 \
    --max_episode_length 1000 \
    --log_interval 50 \
    --eval 1 \
    --vis_type 2
```

# Experiments

## End-to-end PPO
> Baseline PPO with noisy environment

Find code `ppoBaseline/`

### To Run it

- Modify relative paths inside
```
# To train
python -u run.py \
    --exp-config ppo_pointnav_example.yaml \
    --run-type train
# To validate
python -u habitat-api/habitat_baselines/run.py \
    --exp-config habitat-api/habitat_baselines/config/pointnav/ppo_pointnav_example.yaml \
    --run-type eval
```
