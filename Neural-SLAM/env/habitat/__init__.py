# Parts of the code in this file have been borrowed from:
#    https://github.com/facebookresearch/habitat-api

import numpy as np
import torch
from habitat.config.default import get_config as cfg_env
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1

from .pointnav_env import PointNavEnv
from .pointnav_vector_env import VectorEnv
from habitat import get_config as cfg_baseline


def make_env_fn(args, config_env, config_baseline, rank):
    print("-------------- condig_env ---------------")
    print(config_env)
    print("-----------------------------------------")
    dataset = PointNavDatasetV1(config_env.DATASET)
    print("Loading {}".format(config_env.SIMULATOR.SCENE))
    env = PointNavEnv(args=args, rank=rank, config_env=config_env, config_baseline=config_baseline, dataset=dataset)
    env.seed(rank)
    return env


def construct_envs(args):
    env_configs = []
    baseline_configs = []
    args_list = []

    # TODO check params consistency here
    basic_config = cfg_env(config_paths=
                           [args.task_config])

    print("loading scenes ...")
    scenes = PointNavDatasetV1.get_scenes_to_load(basic_config.DATASET)

    if len(scenes) > 0:
        assert len(scenes) >= args.num_processes, (
            "reduce the number of processes as there "
            "aren't enough number of scenes"
        )
        scene_split_size = int(np.floor(len(scenes)/args.num_processes))

    print("using ", args.num_processes, " processes and ", scene_split_size, " scenes per process")

    for i in range(args.num_processes):
        config_env = cfg_env(config_paths=
                             [args.task_config])
        config_env.defrost()

        if len(scenes) > 0:
            config_env.DATASET.CONTENT_SCENES = scenes[
                                                i * scene_split_size: (i + 1) * scene_split_size
                                                ]

        if i < args.num_processes_on_first_gpu:
            gpu_id = 0
        else:
            gpu_id = int((i - args.num_processes_on_first_gpu)
                         // args.num_processes_per_gpu) + args.sim_gpu_id
#         gpu_id = min(torch.cuda.device_count() - 1, gpu_id)
        gpu_id = 0
        config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_id
        config_env.freeze()
        env_configs.append(config_env)

        config_baseline = cfg_baseline()
        baseline_configs.append(config_baseline)

        args_list.append(args)

    envs = VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(
                zip(args_list, env_configs, baseline_configs,
                    range(args.num_processes))
            )
        ),
    )

    # envs = make_env_fn(args_list[0], env_configs[0], config_baseline=baseline_configs[0], rank=42)
    print("returning with environment")

    return envs