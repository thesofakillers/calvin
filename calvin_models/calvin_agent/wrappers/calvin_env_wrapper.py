import logging
import os
from typing import Any, Dict, Tuple, Union

from calvin_agent.datasets.utils.episode_utils import (
    process_depth,
    process_rgb,
    process_state,
)
import gym
import numpy as np
import torch

from calvin_env.envs.play_table_env import get_env
from calvin_env.utils.utils import set_egl_device

logger = logging.getLogger(__name__)


class CalvinEnvWrapper(gym.Wrapper):
    def __init__(
        self,
        dataset_loader,
        urdf_data_dir,
        device,
        egl_dir_path,
        show_gui=False,
        use_egl=True,
        **kwargs,
    ):
        """
        Args:
            dataset_loader: calvin torch dataset
            urdf_data_dir: path to urdf data
            device: torch device to run on
            egl_dir_path: path to directory 'egl_check' with egl libs
            show_gui: whether to show the gui
            use_egl: whether to use egl rendering
            **kwargs: additional arguments
        """
        set_egl_device(device, egl_dir_path)
        env = get_env(
            dataset_loader.abs_datasets_dir,
            urdf_data_dir,
            show_gui=show_gui,
            use_egl=use_egl,
            obs_space=dataset_loader.observation_space,
            **kwargs,
        )
        super(CalvinEnvWrapper, self).__init__(env)
        self.observation_space_keys = dataset_loader.observation_space
        self.transforms = dataset_loader.transforms
        self.proprio_state = dataset_loader.proprio_state
        self.device = device
        self.relative_actions = "rel_actions" in self.observation_space_keys["actions"]
        logger.info(f"Initialized PlayTableEnv for device {self.device}")

    def transform_observation(
        self, obs: Dict[str, Any]
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        state_obs = process_state(
            obs, self.observation_space_keys, self.transforms, self.proprio_state
        )
        rgb_obs = process_rgb(
            obs["rgb_obs"], self.observation_space_keys, self.transforms
        )
        depth_obs = process_depth(
            obs["depth_obs"], self.observation_space_keys, self.transforms
        )

        state_obs["robot_obs"] = state_obs["robot_obs"].to(self.device).unsqueeze(0)
        rgb_obs.update(
            {
                "rgb_obs": {
                    k: v.to(self.device).unsqueeze(0)
                    for k, v in rgb_obs["rgb_obs"].items()
                }
            }
        )
        depth_obs.update(
            {
                "depth_obs": {
                    k: v.to(self.device).unsqueeze(0)
                    for k, v in depth_obs["depth_obs"].items()
                }
            }
        )

        obs_dict: Dict = {
            **rgb_obs,
            **state_obs,
            **depth_obs,
            "robot_obs_raw": torch.from_numpy(obs["robot_obs"]).to(self.device),
        }
        return obs_dict

    def step(
        self, action_tensor: torch.Tensor
    ) -> Tuple[
        Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]], int, bool, Dict
    ]:
        if self.relative_actions:
            action = action_tensor.squeeze().cpu().detach().numpy()
            assert len(action) == 7
        else:
            if action_tensor.shape[-1] == 7:
                slice_ids = [3, 6]
            elif action_tensor.shape[-1] == 8:
                slice_ids = [3, 7]
            else:
                logger.error(
                    "actions are required to have length 8 (for euler angles) or 9 (for quaternions)"
                )
                raise NotImplementedError
            action = np.split(action_tensor.squeeze().cpu().detach().numpy(), slice_ids)
        action[-1] = 1 if action[-1] > 0 else -1
        o, r, d, i = self.env.step(action)

        obs = self.transform_observation(o)
        return obs, r, d, i

    def reset(
        self,
        reset_info: Dict[str, Any] = None,
        batch_idx: int = 0,
        seq_idx: int = 0,
        scene_obs: Any = None,
        robot_obs: Any = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        if reset_info is not None:
            obs = self.env.reset(
                robot_obs=reset_info["robot_obs"][batch_idx, seq_idx],
                scene_obs=reset_info["scene_obs"][batch_idx, seq_idx],
            )
        elif scene_obs is not None or robot_obs is not None:
            obs = self.env.reset(scene_obs=scene_obs, robot_obs=robot_obs)
        else:
            obs = self.env.reset()

        return self.transform_observation(obs)

    def get_info(self):
        return self.env.get_info()

    def get_obs(self):
        obs = self.env.get_obs()
        return self.transform_observation(obs)
