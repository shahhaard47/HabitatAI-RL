import time
from collections import deque
import os
os.environ["OMP_NUM_THREADS"] = "1"
import algo
import sys 
import matplotlib
from tqdm import tqdm
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import gym
import logging
from arguments import get_args
from env import make_vec_envs
from utils.storage import GlobalRolloutStorage, FIFOMemory
from utils.optimization import get_optimizer
from model import RL_Policy, Local_IL_Policy, Neural_SLAM_Module
from habitat.sims.habitat_simulator.actions import HabitatSimActions
# from habitat_baselines.common.tensorboard_utils import TensorboardWriter

if sys.platform == 'darwin':
    matplotlib.use("tkagg")
import matplotlib.pyplot as plt

plt.ion()
fig, ax = plt.subplots(1, 4, figsize=(10, 2.5), facecolor="whitesmoke")

args = get_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def show_gpu_usage():
    print("GPU used ", round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1))
    print("GPU cached ", round(torch.cuda.memory_cached(0) / 1024 ** 3, 1))


def get_local_map_boundaries(agent_loc, local_sizes, full_sizes):
    loc_r, loc_c = agent_loc
    local_w, local_h = local_sizes
    full_w, full_h = full_sizes
    if args.global_downscaling > 1:
        gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
        gx2, gy2 = gx1 + local_w, gy1 + local_h
        if gx1 < 0:
            gx1, gx2 = 0, local_w
        if gx2 > full_w:
            gx1, gx2 = full_w - local_w, full_w
        if gy1 < 0:
            gy1, gy2 = 0, local_h
        if gy2 > full_h:
            gy1, gy2 = full_h - local_h, full_h
    else:
        gx1.gx2, gy1, gy2 = 0, full_w, 0, full_h
    return [gx1, gx2, gy1, gy2]


def get_delta_pose(current_pose, action):
    dpose = np.zeros(current_pose.shape)
    if action == HabitatSimActions.MOVE_FORWARD:
        dpose[0] = 0.25 * np.cos(np.deg2rad(current_pose[2]))
        dpose[1] = -0.25 * np.sin(np.deg2rad(current_pose[2]))
    elif action == HabitatSimActions.TURN_LEFT:
        dpose[2] = np.deg2rad(10)
    elif action == HabitatSimActions.TURN_RIGHT:
        dpose[2] = -np.deg2rad(10)
    return dpose


def main():
    # logging
    log_dir = "{}/models/{}/".format(args.dump_location, args.exp_name)
    dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name)
    tb_dir = log_dir + "tensorboard"
    if not os.path.exists(tb_dir): 
        os.makedirs(tb_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists("{}/images/".format(dump_dir)):
        os.makedirs("{}/images/".format(dump_dir))
    logging.basicConfig(filename=log_dir + 'train.log', level=logging.INFO)
    print("Dumping at {}".format(log_dir))
    print("Arguments starting with ", args)
    logging.info(args)

    # num processes and scenes setup 
    device = args.device = torch.device("cuda:0" if args.cuda else "cpu")   
    num_scenes = args.num_processes
    num_episodes = int(args.num_episodes)
    torch.set_num_threads(1)
    envs = make_vec_envs(args)

    # setting up rewards and losses
    best_cost = float('inf')
    best_slam_loss = 10000000
    costs = deque(maxlen=1000)
    pose_costs = deque(maxlen=1000)
    exp_costs = deque(maxlen=1000)
    
    # Initializing full and local map
    ### Full map consists of 4 channels containing the following:
    ### 1. Obstacle Map
    ### 2. Exploread Area0
    ### 3. Current Agent Location
    ### 4. Past Agent Locations    
    torch.set_grad_enabled(False)
    map_size = args.map_size_cm // args.map_resolution
    full_w, full_h = map_size, map_size
    local_w, local_h = int(full_w / args.global_downscaling), \
                       int(full_h / args.global_downscaling)
    full_map = torch.zeros(num_scenes, 4, full_w, full_h).float().to(device)
    local_map = torch.zeros(num_scenes, 4, local_w, local_h).float().to(device)
    full_pose = torch.zeros(num_scenes, 3).float().to(device)
    local_pose = torch.zeros(num_scenes, 3).float().to(device)
    delta_poses = torch.zeros(num_scenes, 3).float().to(device)
    local_pose_np = local_pose.cpu().numpy()
    full_pose_np = full_pose.cpu().numpy()
    delta_poses_np = delta_poses.cpu().numpy()
    origins = np.zeros((num_scenes, 3))             # Origin of local map
    lmb = np.zeros((num_scenes, 4)).astype(int)     # Local Map Boundaries

    def update_local_map_pose():
        for e in range(num_scenes):
            loc_r, loc_c = [int(full_pose_np[e, 1] * 100.0 / args.map_resolution),
                            int(full_pose_np[e, 0] * 100.0 / args.map_resolution)]
            lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                              (local_w, local_h),
                                              (full_w, full_h))
            origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                          lmb[e][0] * args.map_resolution / 100.0, 0.]
            local_map[e] = full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]]
            local_pose[e] = full_pose[e] - torch.from_numpy(origins[e]).to(device).float()
        local_pose_np = local_pose.cpu().numpy()

    def update_local_visited_area():
        local_map[:, 2, :, :].fill_(0.)
        for e in range(num_scenes):
            r, c = local_pose_np[e, 1], local_pose_np[e, 0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]
            local_map[e, 2:, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.

    def update_full_map_pose():
        for e in range(num_scenes):
            full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]] = local_map[e]
            full_pose[e] = local_pose[e] + torch.from_numpy(origins[e]).to(device).float()
            full_pose_np = full_pose[e].cpu().numpy()

    def init_map_and_pose():
        full_map.fill_(0.)
        full_pose.fill_(0.)
        full_pose[:, :2] = args.map_size_cm / 100.0 / 2.0
        full_pose_np = full_pose.cpu().numpy()
        for e in range(num_scenes):
            loc_r, loc_c = [int(full_pose_np[e, 1] * 100.0 / args.map_resolution),
                            int(full_pose_np[e, 0] * 100.0 / args.map_resolution)]
            full_map[e, 2:, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0
        update_local_map_pose()

    # initializing slam module
    nslam_module = Neural_SLAM_Module(args).to(device)
    slam_optimizer = get_optimizer(nslam_module.parameters(), args.slam_optimizer)
    slam_memory = FIFOMemory(args.slam_memory_size)
    if args.load_slam != "0":
        print("Loading slam {}".format(args.load_slam))
        state_dict = torch.load(args.load_slam,
                                map_location=lambda storage, loc: storage)
        nslam_module.load_state_dict(state_dict)
    if not args.train_slam:
        nslam_module.eval()

    def slam_step(last_obs, obs, delta_poses, build_maps = True):
        _, _, local_map[:, 0, :, :], local_map[:, 1, :, :], _, local_pose = \
            nslam_module(last_obs, obs, delta_poses, local_map[:, 0, :, :],
                         local_map[:, 1, :, :], local_pose, build_maps)
        local_pose_np = local_pose.cpu().numpy()
        update_local_visited_area()
        update_full_map_pose()

    init_map_and_pose()
    obs, infos = envs.reset()
    slam_step(obs, obs, delta_poses)
    envs.update_pose_viz(full_pose)

    start = time.time()
    total_num_steps = -1
    torch.set_grad_enabled(False)
    print("starting episodes")

    # with TensorboardWriter(tb_dir, flush_secs=60) as writer:
        ## why enumerate a range?!! 
    for itr_counter, ep_num in enumerate(range(num_episodes)):
        print("------------------------------------------------------")
        print("Episode", ep_num, itr_counter)
        visimgs = []
        step_bar = tqdm(range(args.max_episode_length))
        for step in step_bar:
            total_num_steps += 1
            l_step = step % args.num_local_steps

            del last_obs
            last_obs = obs.detach()

            gt_action = envs.get_optimal_gt_action().cpu()

            print("gt_action", gt_action)
            obs, rew, done, infos = envs.step(gt_action)
            print("noiseless del_pose: ", infos[0]['sensor_pose'])
            print("pose error: ", infos[0]['pose_err'])

            # Reinitialize variables when episode ends
            if gt_action == HabitatSimActions.STOP or step == args.max_episode_length - 1:
                init_map_and_pose()
                del last_obs
                last_obs = obs.detach()
                print("Reinitialize since at end of episode ")
                obs, infos = envs.reset()

            # Neural SLAM Module
            if args.train_slam:
                # Add frames to memory
                for env_idx in range(num_scenes):
                    delta_poses_np[env_idx] = \
                        get_delta_pose(local_pose_np[env_idx], gt_action[env_idx])
                    env_obs = obs[env_idx].to("cpu")
                    env_poses = torch.from_numpy(delta_poses_np[env_idx]).float().to("cpu")
                    env_gt_fp_projs = torch.from_numpy(np.asarray(
                        infos[env_idx]['fp_proj'])).unsqueeze(0).float().to("cpu")
                    env_gt_fp_explored = torch.from_numpy(np.asarray(
                        infos[env_idx]['fp_explored'])).unsqueeze(0).float().to("cpu")
                    env_gt_pose_err = torch.from_numpy(np.asarray(
                        infos[env_idx]['pose_err'])).float().to("cpu")
                    slam_memory.push(
                        (last_obs[env_idx].cpu(), env_obs, env_poses),
                        (env_gt_fp_projs, env_gt_fp_explored, env_gt_pose_err))

            delta_poses = torch.from_numpy(delta_poses_np).float().to(device)
            slam_step(last_obs, obs, delta_poses)
            envs.update_pose_viz(full_pose)
            if l_step == args.num_local_steps - 1:
                update_local_map_pose()

            ### TRAINING
            torch.set_grad_enabled(True)
            # ------------------------------------------------------------------
            # Train Neural SLAM Module
            slam_overall_loss = 0.0
            if args.train_slam and len(slam_memory) > args.slam_batch_size:
                for _ in range(args.slam_iterations):
                    inputs, outputs = slam_memory.sample(args.slam_batch_size)
                    b_obs_last, b_obs, b_poses = inputs
                    gt_fp_projs, gt_fp_explored, gt_pose_err = outputs

                    b_obs = b_obs.to(device)
                    b_obs_last = b_obs_last.to(device)
                    b_poses = b_poses.to(device)

                    gt_fp_projs = gt_fp_projs.to(device)
                    gt_fp_explored = gt_fp_explored.to(device)
                    gt_pose_err = gt_pose_err.to(device)

                    b_proj_pred, b_fp_exp_pred, _, _, b_pose_err_pred, _ = \
                        nslam_module(b_obs_last, b_obs, b_poses,
                                    None, None, None,
                                    build_maps=False)
                    loss = 0
                    if args.proj_loss_coeff > 0:
                        proj_loss = F.binary_cross_entropy(b_proj_pred,
                                                        gt_fp_projs)
                        costs.append(proj_loss.item())
                        loss += args.proj_loss_coeff * proj_loss

                    if args.exp_loss_coeff > 0:
                        exp_loss = F.binary_cross_entropy(b_fp_exp_pred,
                                                        gt_fp_explored)
                        exp_costs.append(exp_loss.item())
                        loss += args.exp_loss_coeff * exp_loss

                    if args.pose_loss_coeff > 0:
                        pose_loss = torch.nn.MSELoss()(b_pose_err_pred,
                                                    gt_pose_err)
                        pose_costs.append(args.pose_loss_coeff *
                                        pose_loss.item())
                        loss += args.pose_loss_coeff * pose_loss

                    slam_optimizer.zero_grad()
                    loss.backward()
                    slam_optimizer.step()

                    slam_overall_loss += loss

                    del b_obs_last, b_obs, b_poses
                    del gt_fp_projs, gt_fp_explored, gt_pose_err
                    del b_proj_pred, b_fp_exp_pred, b_pose_err_pred

                slam_overall_loss /= args.slam_iterations # mean across iterations
            # Finish Training
            torch.set_grad_enabled(False)

            # ------------------------------------------------------------------
            # Logging

            gettime = lambda: str(datetime.now()).split('.')[0]
            if total_num_steps % args.log_interval == 0:
                end = time.time()
                time_elapsed = time.gmtime(end - start)
                log = " ".join([
                    "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
                    "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
                    gettime(),
                    "num timesteps {},".format(total_num_steps *
                                            num_scenes),
                    "FPS {},".format(int(total_num_steps * num_scenes \
                                        / (end - start)))
                ])

                log += "\n\tLosses:"

                if args.train_slam and len(costs) > 0:
                    log += " ".join([
                        " SLAM Loss overall/proj/exp/pose:"
                        "{:.4f}/{:.4f}/{:.4f}/{:.4f}".format(
                            slam_overall_loss,
                            np.mean(costs),
                            np.mean(exp_costs),
                            np.mean(pose_costs))
                    ])

                print(log)
                logging.info(log)
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Save best models
            if (total_num_steps * num_scenes) % args.save_interval < \
                    num_scenes:

                # Save Neural SLAM Model
                if len(costs) >= 1000 and np.mean(costs) < best_cost \
                        and not args.eval:
                    print("Saved best proj model")
                    best_cost = np.mean(costs)
                    torch.save(nslam_module.state_dict(),
                            os.path.join(log_dir, "model_best_proj.slam"))
                
                if slam_overall_loss < best_slam_loss and not args.eval:
                    print("Saved best overall loss model")
                    best_slam_loss = slam_overall_loss
                    torch.save(nslam_module.state_dict(), 
                            os.path.join(log_dir, "model_best.slam"))

            # writer.add_scalar("proj_loss_slam", np.mean(costs), total_num_steps)
            # writer.add_scalar("exp_loss_slam", np.mean(exp_costs), total_num_steps)
            # writer.add_scalar("pose_loss_slam", np.mean(pose_costs), total_num_steps)
            # writer.add_scalar("SLAM_overall_Loss", slam_overall_loss, total_num_steps)
            # writer.add_scalar("SLAM_best_loss", best_slam_loss, total_num_steps)

            # Save periodic models
            if (total_num_steps * num_scenes) % args.save_periodic < \
                    num_scenes:
                step = total_num_steps * num_scenes
                if args.train_slam:
                    torch.save(nslam_module.state_dict(),
                            os.path.join(dump_dir,
                                            "periodic_{}.slam".format(step)))

            if  l_action == HabitatSimActions.STOP:  # Last episode step
                break
        
        npvis_images = np.array(envs.get_numpy_vis())
        # writer.add_video_from_np_images(
        #     video_name="ep_{}".format(ep_num),
        #     step_idx=total_num_steps,
        #     images=npvis_images,
        #     fps=3
        # )
        envs.reset_numpy_vis()


if __name__ == "__main__":
    main()
