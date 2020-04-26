
python pose_estimation.py --split val --eval 1 \
    --train_global 0 --train_local 0 --train_slam 0 \
    --load_global pretrained_models/model_best.global \
    --load_local pretrained_models/model_best.local \
    --load_slam pretrained_models/model_best.slam \
    --num_processes 1 \
    --task_config ../configs/train_pose_estimation.local.rgbd.yaml \
    --print_images 1 \
    --exp_name exp_first
