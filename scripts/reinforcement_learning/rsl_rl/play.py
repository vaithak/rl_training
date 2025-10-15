# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args
from rl_utils import camera_follow

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
parser.add_argument("--obs_log_csv", type=str, default="policy_observations.csv", help="CSV filename for logging policy observations.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import gymnasium as gym
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import rl_training.tasks  # noqa: F401

class ObservationLogger:
    def __init__(self, env, csv_filename="policy_observations.csv"):
        self.env = env
        self.obs_manager = env.unwrapped.observation_manager
        self.csv_filename_prefix = csv_filename.replace('.csv', '')  # Remove .csv extension if present
        self.num_envs = env.unwrapped.num_envs

        # Track CSV files and writers for each environment
        self.csv_files = {}
        self.csv_writers = {}
        self.header_written = {}

        # Initialize CSV file for each environment
        for env_id in range(self.num_envs):
            filename = f"{self.csv_filename_prefix}_env_{env_id}.csv"
            self.csv_files[env_id] = open(filename, 'w', newline='')
            self.csv_writers[env_id] = None
            self.header_written[env_id] = False

    def _get_column_names(self, group_name, term_names, term_dims, is_concatenated):
        """Generate column names for CSV header"""
        columns = ["timestamp", "dt"]

        for name, dim in zip(term_names, term_dims):
            if dim[0] == 1:
                # Single dimension - just use the term name
                columns.append(name)
            else:
                # Multiple dimensions - add indexed names (term_name_0, term_name_1, ...)
                for i in range(dim[0]):
                    columns.append(f"{name}_{i}")

        return columns

    def log_observations(self):
        """Log policy observations to CSV as time-series data"""
        observations = self.obs_manager.compute()

        # Only log the "policy" group
        if "policy" not in observations:
            return

        group_name = "policy"
        group_data = observations[group_name]

        # Get the shape of individual terms within the group
        term_dims = self.obs_manager.group_obs_term_dim[group_name]

        # Get the list of active term names for the group
        term_names = [term for term in self.obs_manager.active_terms[group_name]]

        # Check if the terms are concatenated into a single tensor
        is_concatenated = self.obs_manager.group_obs_concatenate[group_name]

        # Log data for each environment
        for env_id in range(self.num_envs):
            # Write header if first time for this environment
            if not self.header_written[env_id]:
                columns = self._get_column_names(group_name, term_names, term_dims, is_concatenated)
                self.csv_writers[env_id] = csv.DictWriter(self.csv_files[env_id], fieldnames=columns)
                self.csv_writers[env_id].writeheader()
                self.header_written[env_id] = True

            # Prepare row data
            row = {}
            row["timestamp"] = time.time()
            row["dt"] = self.env.unwrapped.step_dt

            if is_concatenated:
                # If concatenated, we need to manually split the tensor based on term dimensions
                start_idx = 0
                for name, dim in zip(term_names, term_dims):
                    # The dimensions are for a single environment, so we need to account for batch size
                    end_idx = start_idx + dim[0]
                    term_obs = group_data[env_id, start_idx:end_idx]

                    if dim[0] == 1:
                        # Single value
                        row[name] = term_obs[0].item()
                    else:
                        # Multiple values
                        for i, val in enumerate(term_obs):
                            row[f"{name}_{i}"] = val.item()

                    start_idx = end_idx
            else:
                # If not concatenated, the observation group is a dictionary
                for name, term_obs in group_data.items():
                    if term_obs.shape[1] == 1:
                        # Single value
                        row[name] = term_obs[env_id, 0].item()
                    else:
                        # Multiple values
                        for i in range(term_obs.shape[1]):
                            row[f"{name}_{i}"] = term_obs[env_id, i].item()

            # Write row to CSV
            self.csv_writers[env_id].writerow(row)

    def close(self):
        """Close all CSV files"""
        for env_id in range(self.num_envs):
            if env_id in self.csv_files and self.csv_files[env_id]:
                self.csv_files[env_id].close()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 50

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # spawn the robot randomly in the grid (instead of their terrain levels)
    env_cfg.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    # remove random pushing
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.curriculum.command_levels = None

    if args_cli.keyboard:        
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        config = Se2KeyboardCfg(
            v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1]/2,
            v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
            omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
        )
        controller = Se2Keyboard(config)
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device),
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # logger = ObservationLogger(env)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    logger = ObservationLogger(env, csv_filename=args_cli.obs_log_csv)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_onnx(
        policy=policy_nn,
        normalizer=None,
        path=export_model_dir,
        filename="policy.onnx",
    )
    export_policy_as_jit(
        policy=policy_nn,
        normalizer=None,
        path=export_model_dir,
        filename="policy.pt",
    )

    dt = env.unwrapped.step_dt
    # print(dt, "dt")
    # reset environment
    
    obs = env.get_observations()

    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        logger.log_observations()
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)

            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        if args_cli.keyboard:
            camera_follow(env)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    logger.close()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
