# DL Final Project Submission
Code for the Deep Learning project - habitat challenge PointNav CVPR 2020 

<video src="results/Pose estimator/SampleRuns/episodes/1/1/video.mp4" width="320" height="200" controls preload>Sample Run</video>

![Sample Run](results/Pose estimator/SampleRuns/episodes/1/1/video.mp4)

## Team - CS 4803DL-7643A Spring 2020
- Haard Shah
- Akshay Krishnan
- Mason Lilly
- Prateek Vishal

Github Repo: https://github.com/akshay-krishnan/habitat-submission.git 

View our project Report (pdf): [Project_Report___HabitatAI_PointNav-2.pdf](Project_Report___HabitatAI_PointNav-2.pdf)

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

## Contribution - Learn pose estimator explicitly

We do this by first learning an accuracte pose estimator using classical shortest path follower algorithm (TODO name)

> NOTE: more information about options to pass can be found by running `python pose_estimation_optimal.py --help`

Please download `model_best.slam` from [here](https://drive.google.com/file/d/15ufTrfDeF5l-xOlrjsB8BxUx81Q-q6cv/view?usp=sharing) to train or evaluate trained model with gibson dataset. 

After downloading it, replace `<path/to/best>` in the following commands with location of the downloaded model weights.

### To train pose estimator
Code for this can be found here: [Neural-SLAM/pose_estimation_optimal.py](Neural-SLAM/pose_estimation_optimal.py)
Install Habitat Sim(branch stable) and Habitat API(branch habitat-challenge-2020) 
```
cd Neural-SLAM
sudo ln -s /path/to/data data
python pose_estimation_optimal.py \
    --eval 0  --split train_small \
    --train_slam 1 \
    --load_slam <path/to/best>/model_best.slam \
    --num_processes 1 \
    --num_processes_per_gpu 14 \
    --task_config "../configs/train_pose_estimation.local.rgbd.yaml" \
    --exp_name TrainPose1 \
    --print_images 1  --vis_type 2 --d outputs \
    --print_frequency 25 \
    --max_episode_length 1000 \
    --log_interval 50 \
```

### To evaluate pose estimator
Code for this can be found here: [Neural-SLAM/pose_estimation_optimal_eval.py](Neural-SLAM/pose_estimation_optimal_eval.py)

```
cd Neural-SLAM;
python pose_estimation_optimal.py \
    --eval 1  --split val \
    --train_slam 0 \
    --load_slam <path/to/best>/model_best.slam \
    --num_processes 1 \
    --num_processes_per_gpu 14 \
    --task_config "../configs/train_pose_estimation.local.rgbd.yaml" \
    --exp_name EvalPose1 \
    --print_images 1 --vis_type 2 --d outputs \
    --print_frequency 10 \
    --max_episode_length 1000 \
    --log_interval 50 \
```

# Experiments

## End-to-end PPO
> Baseline PPO with noisy environment

Code can be found here [`ppoBaseline/`](`ppoBaseline/`)

### To Run it

> Please modify absolute paths to data inside `../configs/train_pose_estimation.local.rgbd.yaml`

```
# To train
python -u run.py \
    --exp-config ppo_pointnav_example.yaml \
    --run-type train \
    BASE_TASK_CONFIG_PATH ../configs/train_pose_estimation.local.rgbd.yaml
# To validate
python -u run.py \
    --exp-config ppo_pointnav_example.yaml \
    --run-type eval
    BASE_TASK_CONFIG_PATH ../configs/train_pose_estimation.local.rgbd.yaml
```
