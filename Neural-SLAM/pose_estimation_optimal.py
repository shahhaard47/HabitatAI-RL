import time
from collections import deque
import os

os.environ["OMP_NUM_THREADS"] = "1"
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
from habitat_baselines.common.tensorboard_utils import TensorboardWriter

import algo
import sys
import matplotlib

from tqdm import tqdm
from datetime import datetime

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


# should return num_envs * 3 np array
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

    print("---------------------")
    print("Actions")
    print("STOP", HabitatSimActions.STOP)
    print("FORWARD", HabitatSimActions.MOVE_FORWARD)
    print("LEFT", HabitatSimActions.TURN_LEFT)
    print("RIGHT", HabitatSimActions.TURN_RIGHT)

    log_dir = "{}/models/{}/".format(args.dump_location, args.exp_name)
    dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name)
    tb_dir = log_dir + "tensorboard"
    if not os.path.exists(tb_dir): os.makedirs(tb_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists("{}/images/".format(dump_dir)):
        os.makedirs("{}/images/".format(dump_dir))
    logging.basicConfig(
        filename=log_dir + 'train.log',
        level=logging.INFO)
    print("Dumping at {}".format(log_dir))
    print("Arguments starting with ", args)
    logging.info(args)
    device = args.device = torch.device("cuda:0" if args.cuda else "cpu")
    # Logging and loss variables
    num_scenes = args.num_processes
    num_episodes = int(args.num_episodes)

    # setting up rewards and losses
    # policy_loss = 0
    best_cost = float('inf')
    best_slam_loss = 10000000
    costs = deque(maxlen=1000)
    exp_costs = deque(maxlen=1000)
    pose_costs = deque(maxlen=1000)
    l_masks = torch.zeros(num_scenes).float().to(device)
    # best_local_loss = np.inf
    # if args.eval:
    #     traj_lengths = args.max_episode_length // args.num_local_steps
    # l_action_losses = deque(maxlen=1000)
    print("Setup rewards")

    print("starting envrionments ...")
    # Starting environments
    torch.set_num_threads(1)
    envs = make_vec_envs(args)
    obs, infos = envs.reset()
    print("environments reset")

    # show_gpu_usage()
    # Initialize map variables
    ### Full map consists of 4 channels containing the following:
    ### 1. Obstacle Map
    ### 2. Exploread Area
    ### 3. Current Agent Location
    ### 4. Past Agent Locations
    print("creating maps and poses ")
    torch.set_grad_enabled(False)
    # Calculating full and local map sizes
    map_size = args.map_size_cm // args.map_resolution
    full_w, full_h = map_size, map_size
    local_w, local_h = int(full_w / args.global_downscaling), \
                       int(full_h / args.global_downscaling)
    # Initializing full and local map
    full_map = torch.zeros(num_scenes, 4, full_w, full_h).float().to(device)
    local_map = torch.zeros(num_scenes, 4, local_w, local_h).float().to(device)
    # Initial full and local pose
    full_pose = torch.zeros(num_scenes, 3).float().to(device)
    local_pose = torch.zeros(num_scenes, 3).float().to(device)
    # Origin of local map
    origins = np.zeros((num_scenes, 3))
    # Local Map Boundaries
    lmb = np.zeros((num_scenes, 4)).astype(int)
    ### Planner pose inputs has 7 dimensions
    ### 1-3 store continuous global agent location
    ### 4-7 store local map boundaries
    planner_pose_inputs = np.zeros((num_scenes, 7))

    # show_gpu_usage()

    def init_map_and_pose():
        full_map.fill_(0.)
        full_pose.fill_(0.)
        full_pose[:, :2] = args.map_size_cm / 100.0 / 2.0

        full_pose_np = full_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = full_pose_np
        for e in range(num_scenes):
            r, c = full_pose_np[e, 1], full_pose_np[e, 0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]

            full_map[e, 2:, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

            lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                              (local_w, local_h),
                                              (full_w, full_h))

            planner_pose_inputs[e, 3:] = lmb[e]
            origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                          lmb[e][0] * args.map_resolution / 100.0, 0.]
        for e in range(num_scenes):
            local_map[e] = full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]]
            local_pose[e] = full_pose[e] - \
                            torch.from_numpy(origins[e]).to(device).float()
    init_map_and_pose()
    print("maps and poses intialized")

    print("defining architecture")
    # slam
    nslam_module = Neural_SLAM_Module(args).to(device)
    slam_optimizer = get_optimizer(nslam_module.parameters(), args.slam_optimizer)
    slam_memory = FIFOMemory(args.slam_memory_size)

    # # Local policy
    # print("policy observation space", envs.observation_space.spaces['rgb'])
    # print("policy action space ", envs.action_space)
    # l_observation_space = gym.spaces.Box(0, 255,
    #                                      (3,
    #                                       args.frame_width,
    #                                       args.frame_width), dtype='uint8')
    # # todo change this to use envs.observation_space.spaces['rgb'].shape later
    # l_policy = Local_IL_Policy(l_observation_space.shape, envs.action_space.n,
    #                            recurrent=args.use_recurrent_local,
    #                            hidden_size=args.local_hidden_size,
    #                            deterministic=args.use_deterministic_local).to(device)
    # local_optimizer = get_optimizer(l_policy.parameters(), args.local_optimizer)
    # show_gpu_usage()

    print("loading model weights")
    # Loading model
    if args.load_slam != "0":
        print("Loading slam {}".format(args.load_slam))
        state_dict = torch.load(args.load_slam,
                                map_location=lambda storage, loc: storage)
        nslam_module.load_state_dict(state_dict)
    if not args.train_slam:
        nslam_module.eval()

    #     if args.load_local != "0":
    #         print("Loading local {}".format(args.load_local))
    #         state_dict = torch.load(args.load_local,
    #                                 map_location=lambda storage, loc: storage)
    #         l_policy.load_state_dict(state_dict)
    #     if not args.train_local:
    #         l_policy.eval()

    print("predicting first pose and initializing maps")
    # if not (args.use_gt_pose and args.use_gt_map):
        # delta_pose is the expected change in pose when action is applied at
        # the current pose in the absence of noise.
        # initially no action is applied so this is zero.
    delta_poses = torch.from_numpy(np.zeros(local_pose.shape)).float().to(device)
    # initial estimate for local pose and local map from first observation,
    # initialized (zero) pose and map
    _, _, local_map[:, 0, :, :], local_map[:, 1, :, :], _, local_pose = \
        nslam_module(obs, obs, delta_poses, local_map[:, 0, :, :],
                     local_map[:, 1, :, :], local_pose)            
        # if args.use_gt_pose:
        #     # todo update local_pose here
        #     full_pose = envs.get_gt_pose()
        #     for e in range(num_scenes):
        #         local_pose[e] = full_pose[e] - \
        #                         torch.from_numpy(origins[e]).to(device).float()
        # if args.use_gt_map:
        #     full_map = envs.get_gt_map()
        #     for e in range(num_scenes):
        #         local_map[e] = full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]]
    print("slam module returned pose and maps")

    # NOT NEEDED : 4/29 
    local_pose_np = local_pose.cpu().numpy()
    # update local map for each scene - input for planner
    for e in range(num_scenes):
        r, c = local_pose_np[e, 1], local_pose_np[e, 0]
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)]
        local_map[e, 2:, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.

    #     # todo get goal from env here
    global_goals = envs.get_goal_coords().int()

    # Compute planner inputs
    planner_inputs = [{} for e in range(num_scenes)]
    for e, p_input in enumerate(planner_inputs):
        p_input['goal'] = global_goals[e].detach().cpu().numpy()
        p_input['map_pred'] = local_map[e, 0, :, :].detach().cpu().numpy()
        p_input['exp_pred'] = local_map[e, 1, :, :].detach().cpu().numpy()
        p_input['pose_pred'] = planner_pose_inputs[e]

    # Output stores local goals as well as the the ground-truth action
    planner_out = envs.get_short_term_goal(planner_inputs)
    # planner output contains:
    # Distance to short term goal - positive discretized number
    # angle to short term goal -  angle -180 to 180 but in buckets of 5 degrees so multiply by 5 to ge true angle
    # GT action - action to be taken according to planner (int)

    # going to step through the episodes, so cache previous information
    last_obs = obs.detach()
    local_rec_states = torch.zeros(num_scenes, args.local_hidden_size).to(device)
    start = time.time()
    total_num_steps = -1
    torch.set_grad_enabled(False)

    print("starting episodes")
    with TensorboardWriter(
            tb_dir, flush_secs=60
        ) as writer:
        for itr_counter, ep_num in enumerate(range(num_episodes)):
            print("------------------------------------------------------")
            print("Episode", ep_num)

            # if itr_counter >= 20:
            #     print("DONE WE FIXED IT")
            #     die()
            # for step in range(args.max_episode_length):
            visimgs = []
            step_bar = tqdm(range(args.max_episode_length))
            for step in step_bar:
                # print("------------------------------------------------------")
                # print("episode ", ep_num, "step ", step)
                total_num_steps += 1
                l_step = step % args.num_local_steps

                # Local Policy
                # ------------------------------------------------------------------
                # cache previous information
                del last_obs
                last_obs = obs.detach()
                #             if not args.use_optimal_policy and not args.use_shortest_path_gt:
                    #                 local_masks = l_masks
                    #                 local_goals = planner_out[:, :-1].to(device).long()

                    #                 if args.train_local:
                    #                     torch.set_grad_enabled(True)

                    #                 # local policy "step"
                    #                 action, action_prob, local_rec_states = l_policy(
                    #                     obs,
                    #                     local_rec_states,
                    #                     local_masks,
                    #                     extras=local_goals,
                    #                 )

                    #                 if args.train_local:
                    #                     action_target = planner_out[:, -1].long().to(device)
                    #                     # doubt: this is probably wrong? one is action probability and the other is action
                    #                     policy_loss += nn.CrossEntropyLoss()(action_prob, action_target)
                    #                     torch.set_grad_enabled(False)
                    #                 l_action = action.cpu()
                    #             else:
                    #                 if args.use_optimal_policy:
                    #                     l_action = planner_out[:, -1]
                    #                 else:
                    #                     l_action = envs.get_optimal_gt_action()

                l_action = envs.get_optimal_gt_action().cpu()
                # if step > 10:
                #     l_action = torch.tensor([HabitatSimActions.STOP])

                
                
                # ------------------------------------------------------------------
                # ------------------------------------------------------------------
                # Env step
                # print("stepping with action ", l_action)
                # try:
                obs, rew, done, infos = envs.step(l_action)


                # ------------------------------------------------------------------
                # Reinitialize variables when episode ends
                # doubt what if episode ends before max_episode_length?
                # maybe add (or done ) here?
                if l_action == HabitatSimActions.STOP or step == args.max_episode_length - 1:
                    print("l_action", l_action)
                    init_map_and_pose()
                    del last_obs
                    last_obs = obs.detach()
                    print("Reinitialize since at end of episode ") 
                    obs, infos = envs.reset()


                # except:
                #     print("can't do that")
                #     print(l_action)
                #     init_map_and_pose()
                #     del last_obs
                #     last_obs = obs.detach()
                #     print("Reinitialize since at end of episode ")
                #     break
                # step_bar.set_description("rew, done, info-sensor_pose, pose_err (stepping) {}, {}, {}, {}".format(rew, done, infos[0]['sensor_pose'], infos[0]['pose_err']))
                if total_num_steps % args.log_interval == 0 and False:
                    print("rew, done, info-sensor_pose, pose_err after stepping ", rew, done, infos[0]['sensor_pose'],
                    infos[0]['pose_err'])
                # l_masks = torch.FloatTensor([0 if x else 1
                #                              for x in done]).to(device)

                # ------------------------------------------------------------------
                # # ------------------------------------------------------------------
                # # Reinitialize variables when episode ends
                # # doubt what if episode ends before max_episode_length?
                # # maybe add (or done ) here?
                # if step == args.max_episode_length - 1 or l_action == HabitatSimActions.STOP:  # Last episode step
                #     init_map_and_pose()
                #     del last_obs
                #     last_obs = obs.detach()
                #     print("Reinitialize since at end of episode ")
                #     break

                # ------------------------------------------------------------------
                # ------------------------------------------------------------------
                # Neural SLAM Module
                delta_poses_np = np.zeros(local_pose_np.shape)
                if args.train_slam:
                    # Add frames to memory
                    for env_idx in range(num_scenes):
                        delta_poses_np[env_idx] = get_delta_pose(local_pose_np[env_idx], l_action[env_idx])
                        env_obs = obs[env_idx].to("cpu")
                        env_poses = torch.from_numpy(np.asarray(
                            delta_poses_np[env_idx]
                        )).float().to("cpu")
                        env_gt_fp_projs = torch.from_numpy(np.asarray(
                            infos[env_idx]['fp_proj']
                        )).unsqueeze(0).float().to("cpu")
                        env_gt_fp_explored = torch.from_numpy(np.asarray(
                            infos[env_idx]['fp_explored']
                        )).unsqueeze(0).float().to("cpu")
                        # TODO change pose err here
                        env_gt_pose_err = torch.from_numpy(np.asarray(
                            infos[env_idx]['pose_err']
                        )).float().to("cpu")
                        slam_memory.push(
                            (last_obs[env_idx].cpu(), env_obs, env_poses),
                            (env_gt_fp_projs, env_gt_fp_explored, env_gt_pose_err))
                delta_poses = torch.from_numpy(delta_poses_np).float().to(device)
                # print("delta pose from SLAM ", delta_poses)
                _, _, local_map[:, 0, :, :], local_map[:, 1, :, :], _, local_pose = \
                        nslam_module(last_obs, obs, delta_poses, local_map[:, 0, :, :],
                                    local_map[:, 1, :, :], local_pose, build_maps=True)
                # print("updated local pose from SLAM ", local_pose)
                # if args.use_gt_pose:
                #     # todo update local_pose here
                #     full_pose = envs.get_gt_pose()
                #     for e in range(num_scenes):
                #         local_pose[e] = full_pose[e] - \
                #                         torch.from_numpy(origins[e]).to(device).float()
                #     print("updated local pose from gt ", local_pose)
                # if args.use_gt_map:
                #     full_map = envs.get_gt_map()
                #     for e in range(num_scenes):
                #         local_map[e] = full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]]
                #     print("updated local map from gt")

                local_pose_np = local_pose.cpu().numpy()
                planner_pose_inputs[:, :3] = local_pose_np + origins
                local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel
                for e in range(num_scenes):
                    r, c = local_pose_np[e, 1], local_pose_np[e, 0]
                    loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                    int(c * 100.0 / args.map_resolution)]
                    local_map[e, 2:, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.

                planner_inputs = [{} for e in range(num_scenes)]
                for e, p_input in enumerate(planner_inputs):
                    p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
                    p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
                    p_input['pose_pred'] = planner_pose_inputs[e]
                    p_input['goal'] = global_goals[e].cpu().numpy()
                
                planner_out = envs.get_short_term_goal(planner_inputs)


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

                        if args.train_slam:
                            slam_optimizer.zero_grad()
                            loss.backward()
                            slam_optimizer.step()

                        slam_overall_loss += loss

                        del b_obs_last, b_obs, b_poses
                        del gt_fp_projs, gt_fp_explored, gt_pose_err
                        del b_proj_pred, b_fp_exp_pred, b_pose_err_pred

                    slam_overall_loss /= args.slam_iterations # mean across iterations
                # ------------------------------------------------------------------

                # ------------------------------------------------------------------
                # Train Local Policy
                    # if (l_step + 1) % args.local_policy_update_freq == 0 \
                    #         and args.train_local:
                    #     local_optimizer.zero_grad()
                    #     policy_loss.backward()
                    #     local_optimizer.step()
                    #     l_action_losses.append(policy_loss.item())
                    #     policy_loss = 0
                    #     local_rec_states = local_rec_states.detach_()
                # ------------------------------------------------------------------

                # Finish Training
                torch.set_grad_enabled(False)
                # ------------------------------------------------------------------

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

                    # if args.train_local and len(l_action_losses) > 0:
                    #     log += " ".join([
                    #         " Local Loss:",
                    #         "{:.3f},".format(
                    #             np.mean(l_action_losses))
                    #     ])

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

                    # Save Local Policy Model
                    # if len(l_action_losses) >= 100 and \
                    #         (np.mean(l_action_losses) <= best_local_loss) \
                    #         and not args.eval:
                    #     torch.save(l_policy.state_dict(),
                    #                os.path.join(log_dir, "model_best.local"))
                    #
                    #     best_local_loss = np.mean(l_action_losses)
                writer.add_scalar("proj_loss_slam", np.mean(costs), total_num_steps)
                writer.add_scalar("exp_loss_slam", np.mean(exp_costs), total_num_steps)
                writer.add_scalar("pose_loss_slam", np.mean(pose_costs), total_num_steps)
                writer.add_scalar("SLAM_overall_Loss", slam_overall_loss, total_num_steps)
                writer.add_scalar("SLAM_best_loss", best_slam_loss, total_num_steps)
                # Save periodic models
                if (total_num_steps * num_scenes) % args.save_periodic < \
                        num_scenes:
                    step = total_num_steps * num_scenes
                    if args.train_slam:
                        torch.save(nslam_module.state_dict(),
                                os.path.join(dump_dir,
                                                "periodic_{}.slam".format(step)))
                    # if args.train_local:
                    #     torch.save(l_policy.state_dict(),
                    #                os.path.join(dump_dir,
                    #                             "periodic_{}.local".format(step)))
                # ------------------------------------------------------------------

                if  l_action == HabitatSimActions.STOP:  # Last episode step
                    break
            
            npvis_images = np.array(envs.get_numpy_vis())
            writer.add_video_from_np_images(
                video_name="ep_{}".format(ep_num),
                step_idx=total_num_steps,
                images=npvis_images,
                fps=3
            )
            envs.reset_numpy_vis()

    # Print and save model performance numbers during evaluation
    if args.eval:
        logfile = open("{}/explored_area.txt".format(dump_dir), "w+")
        for e in range(num_scenes):
            for i in range(explored_area_log[e].shape[0]):
                logfile.write(str(explored_area_log[e, i]) + "\n")
                logfile.flush()

        logfile.close()

        logfile = open("{}/explored_ratio.txt".format(dump_dir), "w+")
        for e in range(num_scenes):
            for i in range(explored_ratio_log[e].shape[0]):
                logfile.write(str(explored_ratio_log[e, i]) + "\n")
                logfile.flush()

        logfile.close()

        log = "Final Exp Area: \n"
        for i in range(explored_area_log.shape[2]):
            log += "{:.5f}, ".format(
                np.mean(explored_area_log[:, :, i]))

        log += "\nFinal Exp Ratio: \n"
        for i in range(explored_ratio_log.shape[2]):
            log += "{:.5f}, ".format(
                np.mean(explored_ratio_log[:, :, i]))

        print(log)
        logging.info(log)


if __name__ == "__main__":
    main()
