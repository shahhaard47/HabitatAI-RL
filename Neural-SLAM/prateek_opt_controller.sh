# Command to run pose_estimation_optimal.py (contains Prateek's shortest path follower)

# TRAIN
# python pose_estimation_optimal.py \
#     --split train_small \
#     --train_slam 1 \
#     --load_slam pretrained_models/model_best.slam \
#     --num_processes 1 \
#     --num_processes_per_gpu 12 \
#     --task_config "/home/haardshah/habitat-submission/configs/train_pose_estimation.local.rgbd.yaml" \
#     --exp_name optimalController \
#     --print_images 1 \
#     --max_episode_length 1000 \
#     --log_interval 100 \
#     --eval 1

# EVAL
python pose_estimation_optimal.py \
    --split val \
    --eval 1 \
    --load_global pretrained_models/model_best.global --train_global 0 \
    --load_local pretrained_models/model_best.local  --train_local 0 \
    --load_slam pretrained_models/model_best.slam  --train_slam 0 \
    --num_processes 1 \
    --num_processes_per_gpu 11 \
    --task_config "/home/haardshah/habitat-submission/configs/train_pose_estimation.local.rgbd.yaml" \
    --exp_name observations \
    --print_images 1 \
    --seed 12


# python  main.py --split val_mt --eval 1 \
# --auto_gpu_config 0 -n 14 --num_episodes 71 --num_processes_per_gpu 7 \
# --load_global pretrained_models/model_best.global --train_global 0 \
# --load_local pretrained_models/model_best.local  --train_local 0 \
# --load_slam pretrained_models/model_best.slam  --train_slam 0