import math
import os
import pickle
import sys

import gym
import matplotlib
import numpy as np
import quaternion
import skimage.morphology
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms

if sys.platform == 'darwin':
    matplotlib.use("tkagg")
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import habitat
from habitat import logger

from env.utils.map_builder import MapBuilder
from env.utils.fmm_planner import FMMPlanner

from env.habitat.utils import pose as pu
from env.habitat.utils import visualizations as vu
from env.habitat.utils.supervision import HabitatMaps
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from model import get_grid


def _preprocess_depth(depth):
    depth = depth[:, :, 0]*1
    mask2 = depth > 0.99
    depth[mask2] = 0.

    for i in range(depth.shape[1]):
        depth[:,i][depth[:,i] == 0.] = depth[:,i].max()

    mask1 = depth == 0
    depth[mask1] = np.NaN
    depth = depth*1000.
    return depth

# Extended environment that keeps track of ground truth and map
class PointNavEnv(habitat.RLEnv):
    # constructor: save args
    # doubt: where are config baseline and dataset used? 
    def __init__(self, args, rank, config_env, config_baseline, dataset):
        if args.visualize:
            plt.ion()
        if args.print_images or args.visualize:
            self.figure, self.ax = plt.subplots(1,2, figsize=(6*16/9, 6),
                                                facecolor="whitesmoke",
                                                num="Thread {}".format(rank))
        self.args = args
        self.num_actions = 3
        self.dt = 10

        super().__init__(config_env)

        self.mapper = self.build_mapper()
        self.episode_no = 0
        self.res = transforms.Compose([transforms.ToPILImage(),
                    transforms.Resize((args.frame_height, args.frame_width),
                                      interpolation = Image.NEAREST)])
        self.scene_name = None
        self.maps_dict = {}

    # shuffle the order of episodes once in a while
    def randomize_env(self):
        self._env._episode_iterator._shuffle_iterator()

    # saves all "states - positions and rotations" that were collected in this trajectory
    def save_trajectory_data(self):
        if "replica" in self.scene_name:
            folder = self.args.save_trajectory_data + "/" + \
                        self.scene_name.split("/")[-3]+"/"
        else:
            folder = self.args.save_trajectory_data + "/" + \
                        self.scene_name.split("/")[-1].split(".")[0]+"/"
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = folder+str(self.episode_no)+".txt"
        with open(filepath, "w+") as f:
            f.write(self.scene_name+"\n")
            for state in self.trajectory_states:
                f.write(str(state)+"\n")
            f.flush()


    def save_position(self):
        self.agent_state = self._env.sim.get_agent_state()
        self.trajectory_states.append([self.agent_state.position,
                                       self.agent_state.rotation])


    # setup of environment
    # initialize states, trajectory of states, episode counter, 
    # timestep, rgb, depth, GT_explorable_map, prev_exp_area, 
    # current map, current and past positions in the map, fp map, 
    # doubt: what is collision map?
    def reset(self):
        args = self.args    # doubt: why is this cached? 
        self.episode_no += 1
        self.timestep = 0
        self._previous_action = None
        # list of x, y, theta
        self.trajectory_states = []

        # TODO what is randomize_env()? 
        if args.randomize_env_every > 0:
            if np.mod(self.episode_no, args.randomize_env_every) == 0:
                self.randomize_env()

        # get obs from env after reset
        # Get Ground Truth Map - explorable map for the given map size
        # set explored area to 0
        self.explorable_map = None
        while self.explorable_map is None:
            obs = super().reset()
            full_map_size = args.map_size_cm//args.map_resolution
            self.explorable_map = self._get_gt_map(full_map_size)
        self.prev_explored_area = 0.

        # Preprocess observations
        # self.obs is rgb to 0-255
        # if the req frame width is not the same as env frame width, resize
        # state is the input to the cnn
        # get depth as well
        rgb = obs['rgb'].astype(np.uint8)
        self.obs = rgb # For visualization
        if self.args.frame_width != self.args.env_frame_width:
            rgb = np.asarray(self.res(rgb))
        state = rgb.transpose(2, 0, 1)
        depth = _preprocess_depth(obs['depth'])

        # Initialize map that is being built and pose in the map
        self.map_size_cm = args.map_size_cm
        self.mapper.reset_map(self.map_size_cm)
        # pose in the map grid coords - gt and actual start at mid of map
        self.curr_loc = [self.map_size_cm/100.0/2.0,
                         self.map_size_cm/100.0/2.0, 0.]
        self.curr_loc_gt = self.curr_loc
        self.last_loc_gt = self.curr_loc_gt
        self.last_loc = self.curr_loc
        # position in simulator coordinates
        self.last_sim_location = self.get_sim_location()

        # Convert pose to cm and degrees for mapper
        mapper_gt_pose = (self.curr_loc_gt[0]*100.0,
                          self.curr_loc_gt[1]*100.0,
                          np.deg2rad(self.curr_loc_gt[2]))

        # Update map, map in this frame, and explored area in this frame based on depth
        fp_proj, self.map, fp_explored, self.explored_map = \
            self.mapper.update_map(depth, mapper_gt_pose)

        # Initialize variables
        self.scene_name = self.habitat_env.sim.config.SCENE
        self.visited = np.zeros(self.map.shape)
        self.visited_vis = np.zeros(self.map.shape)
        self.visited_gt = np.zeros(self.map.shape)
        self.collison_map = np.zeros(self.map.shape)
        self.col_width = 1

        # Set info
        self.info = {
            'time': self.timestep,
            'fp_proj': fp_proj,
            'fp_explored': fp_explored,
            'sensor_pose': [0., 0., 0.],
            'pose_err': [0., 0., 0.]
        }
        self.save_position()
        return state, self.info

    def get_noiseless_map_dpose(self, current_pose, action):
        dx, dy, do = 0.0, 0.0, 0.0
        if action == HabitatSimActions.MOVE_FORWARD:
            dx = 0.25*np.cos(current_pose[2])
            dy = 0.25*np.sin(current_pose[2])
        elif action == HabitatSimActions.TURN_LEFT:
            do = -np.deg2rad(10)
        elif action == HabitatSimActions.TURN_RIGHT:
            do = np.deg2rad(10)
        return dx, dy, do

    # update the positions, maps, based on action
    def step(self, action):
        args = self.args
        self.timestep += 1

        # update previous locations and actions
        self.last_loc = np.copy(self.curr_loc)
        self.last_loc_gt = np.copy(self.curr_loc_gt)
        self._previous_action = action

        # get data from simulator
        obs, rew, done, info = super().step(action)

        # Preprocess observations - same as in reset
        rgb = obs['rgb'].astype(np.uint8)
        self.obs = rgb # For visualization
        if self.args.frame_width != self.args.env_frame_width:
            rgb = np.asarray(self.res(rgb))
        state = rgb.transpose(2, 0, 1)
        depth = _preprocess_depth(obs['depth'])
 
        # ground-truth pose and predicted pose
        # gt pose change from sim
        dx_gt, dy_gt, do_gt = self.get_gt_pose_change()
        dx_base, dy_base, do_base = self.get_noiseless_map_dpose(self.curr_loc, action)
        self.curr_loc = pu.get_new_pose(self.curr_loc,
                               (dx_base, dy_base, do_base))
        self.curr_loc_gt = pu.get_new_pose(self.curr_loc_gt,
                               (dx_gt, dy_gt, do_gt))
        # Convert pose to cm and degrees for mapper
        mapper_gt_pose = (self.curr_loc_gt[0]*100.0,
                          self.curr_loc_gt[1]*100.0,
                          np.deg2rad(self.curr_loc_gt[2]))

        # Update ground_truth map and explored area
        fp_proj, self.map, fp_explored, self.explored_map = \
                self.mapper.update_map(depth, mapper_gt_pose)

        # Update collision map
        # we should use estimated pose for this
        if action == 1:
            x1, y1, t1 = self.last_loc
            x2, y2, t2 = self.curr_loc
            if abs(x1 - x2)< 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                self.col_width = min(self.col_width, 9)
            else:
                self.col_width = 1

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < args.collision_threshold: #Collision
                length = 2
                width = self.col_width
                buf = 3
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05*((i+buf) * np.cos(np.deg2rad(t1)) + \
                                        (j-width//2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05*((i+buf) * np.sin(np.deg2rad(t1)) - \
                                        (j-width//2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(r*100/args.map_resolution), \
                               int(c*100/args.map_resolution)
                        [r, c] = pu.threshold_poses([r, c],
                                    self.collison_map.shape)
                        self.collison_map[r,c] = 1

        # Set info
        self.info['time'] = self.timestep
        self.info['fp_proj'] = fp_proj
        self.info['fp_explored']= fp_explored
        self.info['sensor_pose'] = [dx_base, dy_base, do_base]        
        self.info['pose_err'] = [dx_gt - dx_base,
                                 dy_gt - dy_base,
                                 do_gt - do_base]


        if self.timestep%args.num_local_steps==0:
            area, ratio = self.get_global_reward()
            self.info['exp_reward'] = area
            self.info['exp_ratio'] = ratio
        else:
            self.info['exp_reward'] = None
            self.info['exp_ratio'] = None

        self.save_position()

        if self.info['time'] >= args.max_episode_length:
            done = True
            if self.args.save_trajectory_data != "0":
                self.save_trajectory_data()
        else:
            done = False
        return state, rew, done, self.info


    def get_reward_range(self):
        # This function is not used, Habitat-RLEnv requires this function
        return (0., 1.0)

    def get_reward(self, observations):
        # This function is not used, Habitat-RLEnv requires this function
        return 0.

    def get_global_reward(self):
        curr_explored = self.explored_map*self.explorable_map
        curr_explored_area = curr_explored.sum()

        reward_scale = self.explorable_map.sum()
        m_reward = (curr_explored_area - self.prev_explored_area)*1.
        m_ratio = m_reward/reward_scale
        m_reward = m_reward * 25./10000. # converting to m^2
        self.prev_explored_area = curr_explored_area

        m_reward *= 0.02 # Reward Scaling

        return m_reward, m_ratio

    def get_done(self, observations):
        # This function is not used, Habitat-RLEnv requires this function
        return False

    def get_info(self, observations):
        # This function is not used, Habitat-RLEnv requires this function
        info = {}
        return info

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)

    def get_spaces(self):
        return self.observation_space, self.action_space

    def build_mapper(self):
        params = {}
        params['frame_width'] = self.args.env_frame_width
        params['frame_height'] = self.args.env_frame_height
        params['fov'] =  self.args.hfov
        params['resolution'] = self.args.map_resolution
        params['map_size_cm'] = self.args.map_size_cm
        params['agent_min_z'] = 25
        params['agent_max_z'] = 150
        params['agent_height'] = self.args.camera_height * 100
        params['agent_view_angle'] = 0
        params['du_scale'] = self.args.du_scale
        params['vision_range'] = self.args.vision_range
        params['visualize'] = self.args.visualize
        params['obs_threshold'] = self.args.obs_threshold
        self.selem = skimage.morphology.disk(self.args.obstacle_boundary /
                                             self.args.map_resolution)
        mapper = MapBuilder(params)
        return mapper


    def get_sim_location(self):
        agent_state = super().habitat_env.sim.get_agent_state(0)
        x = -agent_state.position[2]
        y = -agent_state.position[0]
        axis = quaternion.as_euler_angles(agent_state.rotation)[0]
        if (axis%(2*np.pi)) < 0.1 or (axis%(2*np.pi)) > 2*np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_state.rotation)[1]
        else:
            o = 2*np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o


    def get_gt_pose_change(self):
        curr_sim_pose = self.get_sim_location()
        dx, dy, do = pu.get_rel_pose_change(curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose
        return dx, dy, do

    def _get_gt_map(self, full_map_size):
        self.scene_name = self.habitat_env.sim.config.SCENE
        logger.error('Computing map for %s', self.scene_name)

        # Get map in habitat simulator coordinates
        self.map_obj = HabitatMaps(self.habitat_env)
        if self.map_obj.size[0] < 1 or self.map_obj.size[1] < 1:
            logger.error("Invalid map: {}/{}".format(
                            self.scene_name, self.episode_no))
            return None

        agent_y = self._env.sim.get_agent_state().position.tolist()[1]*100.
        sim_map = self.map_obj.get_map(agent_y, -50., 50.0)

        sim_map[sim_map > 0] = 1.

        # Transform the map to align with the agent
        min_x, min_y = self.map_obj.origin/100.0
        x, y, o = self.get_sim_location()
        x, y = -x - min_x, -y - min_y
        range_x, range_y = self.map_obj.max/100. - self.map_obj.origin/100.

        map_size = sim_map.shape
        scale = 2.
        grid_size = int(scale*max(map_size))
        grid_map = np.zeros((grid_size, grid_size))

        grid_map[(grid_size - map_size[0])//2:
                 (grid_size - map_size[0])//2 + map_size[0],
                 (grid_size - map_size[1])//2:
                 (grid_size - map_size[1])//2 + map_size[1]] = sim_map

        if map_size[0] > map_size[1]:
            st = torch.tensor([[
                    (x - range_x/2.) * 2. / (range_x * scale) \
                             * map_size[1] * 1. / map_size[0],
                    (y - range_y/2.) * 2. / (range_y * scale),
                    180.0 + np.rad2deg(o)
                ]])

        else:
            st = torch.tensor([[
                    (x - range_x/2.) * 2. / (range_x * scale),
                    (y - range_y/2.) * 2. / (range_y * scale) \
                            * map_size[0] * 1. / map_size[1],
                    180.0 + np.rad2deg(o)
                ]])

        rot_mat, trans_mat = get_grid(st, (1, 1,
            grid_size, grid_size), torch.device("cpu"))

        grid_map = torch.from_numpy(grid_map).float()
        grid_map = grid_map.unsqueeze(0).unsqueeze(0)
        translated = F.grid_sample(grid_map, trans_mat)
        rotated = F.grid_sample(translated, rot_mat)

        episode_map = torch.zeros((full_map_size, full_map_size)).float()
        if full_map_size > grid_size:
            episode_map[(full_map_size - grid_size)//2:
                        (full_map_size - grid_size)//2 + grid_size,
                        (full_map_size - grid_size)//2:
                        (full_map_size - grid_size)//2 + grid_size] = \
                                rotated[0,0]
        else:
            episode_map = rotated[0,0,
                              (grid_size - full_map_size)//2:
                              (grid_size - full_map_size)//2 + full_map_size,
                              (grid_size - full_map_size)//2:
                              (grid_size - full_map_size)//2 + full_map_size]

        episode_map = episode_map.numpy()
        episode_map[episode_map > 0] = 1.

        return episode_map


    def _get_stg(self, grid, explored, start, goal, planning_window):

        [gx1, gx2, gy1, gy2] = planning_window

        x1 = min(start[0], goal[0])
        x2 = max(start[0], goal[0])
        y1 = min(start[1], goal[1])
        y2 = max(start[1], goal[1])
        dist = pu.get_l2_distance(goal[0], start[0], goal[1], start[1])
        buf = max(20., dist)
        x1 = max(1, int(x1 - buf))
        x2 = min(grid.shape[0]-1, int(x2 + buf))
        y1 = max(1, int(y1 - buf))
        y2 = min(grid.shape[1]-1, int(y2 + buf))

        rows = explored.sum(1)
        rows[rows>0] = 1
        ex1 = np.argmax(rows)
        ex2 = len(rows) - np.argmax(np.flip(rows))

        cols = explored.sum(0)
        cols[cols>0] = 1
        ey1 = np.argmax(cols)
        ey2 = len(cols) - np.argmax(np.flip(cols))

        ex1 = min(int(start[0]) - 2, ex1)
        ex2 = max(int(start[0]) + 2, ex2)
        ey1 = min(int(start[1]) - 2, ey1)
        ey2 = max(int(start[1]) + 2, ey2)

        x1 = max(x1, ex1)
        x2 = min(x2, ex2)
        y1 = max(y1, ey1)
        y2 = min(y2, ey2)

        traversible = skimage.morphology.binary_dilation(
                        grid[x1:x2, y1:y2],
                        self.selem) != True
        traversible[self.collison_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1

        traversible[int(start[0]-x1)-1:int(start[0]-x1)+2,
                    int(start[1]-y1)-1:int(start[1]-y1)+2] = 1

        if goal[0]-2 > x1 and goal[0]+3 < x2\
            and goal[1]-2 > y1 and goal[1]+3 < y2:
            traversible[int(goal[0]-x1)-2:int(goal[0]-x1)+3,
                    int(goal[1]-y1)-2:int(goal[1]-y1)+3] = 1
        else:
            goal[0] = min(max(x1, goal[0]), x2)
            goal[1] = min(max(y1, goal[1]), y2)

        def add_boundary(mat):
            h, w = mat.shape
            new_mat = np.ones((h+2,w+2))
            new_mat[1:h+1,1:w+1] = mat
            return new_mat

        traversible = add_boundary(traversible)

        planner = FMMPlanner(traversible, 360//self.dt)

        reachable = planner.set_goal([goal[1]-y1+1, goal[0]-x1+1])

        stg_x, stg_y = start[0] - x1 + 1, start[1] - y1 + 1
        for i in range(self.args.short_goal_dist):
            stg_x, stg_y, replan = planner.get_short_term_goal([stg_x, stg_y])
        if replan:
            stg_x, stg_y = start[0], start[1]
        else:
            stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        return (stg_x, stg_y)


    def _get_gt_action(self, grid, start, goal, planning_window, start_o):
        [gx1, gx2, gy1, gy2] = planning_window

        x1 = min(start[0], goal[0])
        x2 = max(start[0], goal[0])
        y1 = min(start[1], goal[1])
        y2 = max(start[1], goal[1])
        dist = pu.get_l2_distance(goal[0], start[0], goal[1], start[1])
        buf = max(5., dist)
        x1 = max(0, int(x1 - buf))
        x2 = min(grid.shape[0], int(x2 + buf))
        y1 = max(0, int(y1 - buf))
        y2 = min(grid.shape[1], int(y2 + buf))

        path_found = False
        goal_r = 0
        while not path_found:
            traversible = skimage.morphology.binary_dilation(
                            grid[gx1:gx2, gy1:gy2][x1:x2, y1:y2],
                            self.selem) != True
            traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1
            traversible[int(start[0]-x1)-1:int(start[0]-x1)+2,
                        int(start[1]-y1)-1:int(start[1]-y1)+2] = 1
            traversible[int(goal[0]-x1)-goal_r:int(goal[0]-x1)+goal_r+1,
                        int(goal[1]-y1)-goal_r:int(goal[1]-y1)+goal_r+1] = 1
            scale = 1
            planner = FMMPlanner(traversible, 360//self.dt, scale)

            reachable = planner.set_goal([goal[1]-y1, goal[0]-x1])

            stg_x_gt, stg_y_gt = start[0] - x1, start[1] - y1
            for i in range(1):
                stg_x_gt, stg_y_gt, replan = \
                        planner.get_short_term_goal([stg_x_gt, stg_y_gt])

            if replan and buf < 100.:
                buf = 2*buf
                x1 = max(0, int(x1 - buf))
                x2 = min(grid.shape[0], int(x2 + buf))
                y1 = max(0, int(y1 - buf))
                y2 = min(grid.shape[1], int(y2 + buf))
            elif replan and goal_r < 50:
                goal_r += 1
            else:
                path_found = True

        stg_x_gt, stg_y_gt = stg_x_gt + x1, stg_y_gt + y1
        angle_st_goal = math.degrees(math.atan2(stg_x_gt - start[0],
                                                stg_y_gt - start[1]))
        angle_agent = (start_o)%360.0
        if angle_agent > 180:
            angle_agent -= 360

        relative_angle = (angle_agent - angle_st_goal)%360.0
        if relative_angle > 180:
            relative_angle -= 360

        if relative_angle > 15.:
            gt_action = 1
        elif relative_angle < -15.:
            gt_action = 0
        else:
            gt_action = 2

        return gt_action