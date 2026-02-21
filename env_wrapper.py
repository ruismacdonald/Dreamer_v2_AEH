import gym
import sys
import torch
import numpy as np
from PIL import Image


class DeepMindControl:
    def __init__(self, name, seed, random_targets=None, size=(64, 64), camera=None):

        domain, task = name.split("-", 1)
        if domain == "cup":  # Only domain with multiple words.
            domain = "ball_in_cup"
        if isinstance(domain, str):
            from dm_control import suite

            task_kwargs = {"random": seed}
            if random_targets is not None:
                task_kwargs["random_targets"] = random_targets

            self._env = suite.load(domain, task, task_kwargs=task_kwargs)
        else:
            assert task is None
            self._env = domain()
        self._size = size
        if camera is None:
            camera = dict(quadruped=2).get(domain, 0)
        self._camera = camera

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            spaces[key] = gym.spaces.Box(-np.inf, np.inf, value.shape, dtype=np.float32)
        spaces["image"] = gym.spaces.Box(0, 255, (3,) + self._size, dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        time_step = self._env.step(action)
        obs = dict()
        obs["image"] = self.render().transpose(2, 0, 1).copy()
        reward = time_step.reward or 0
        done = time_step.last()
        info = {"discount": np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self):
        _ = self._env.reset()
        obs = dict()
        obs["image"] = self.render().transpose(2, 0, 1).copy()
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)


class DMCLoCA(DeepMindControl):
    def __init__(
        self,
        name,
        seed,
        random_targets=None,
        size=(64, 64),
        camera=None,
        loca_phase="phase_1",
        loca_mode="train",
        one_way_wall_radius=0.1,
    ):
        super().__init__(name, seed, random_targets=random_targets, size=size, camera=camera)
        self._loca_phase = loca_phase
        self._loca_mode = loca_mode

        if self._loca_phase != "phase_1":
            self._env._task.switch_loca_task()

        self._actuators_length = np.array(
            [
                self._env._physics.named.model.body_pos["hand", "x"],
                self._env._physics.named.model.body_pos["finger", "x"],
            ]
        )
        self._one_way_wall_radius = one_way_wall_radius

    def get_target_1_pos(self):
        return self._env._physics.named.data.geom_xpos["target_1", :2]

    def get_finger_pos(self):
        return self._env._physics.named.data.geom_xpos["finger", :2]

    def check_inside_one_way_wall(self):
        target_1_pos = self.get_target_1_pos()
        finger_pos = self.get_finger_pos()
        return np.linalg.norm(finger_pos - target_1_pos) <= self._one_way_wall_radius

    def is_phase_2(self):
        return self._loca_phase == "phase_2"

    def step(self, action):
        prev_inside_wall = self.check_inside_one_way_wall()
        prev_physics_state = self._env.physics.get_state()

        time_step = self._env.step(action)
        reward = time_step.reward
        done = time_step.last()
        info = {"discount": np.array(time_step.discount, np.float32)}

        if prev_inside_wall and (not self.check_inside_one_way_wall()):
            with self._env._physics.reset_context():
                self._env.physics.set_state(prev_physics_state)

        obs = {"image": self.render().transpose(2, 0, 1).copy()}

        return obs, reward, done, info

    def sample_in_one_way_wall(self):
        def sample_xy():
            x = np.random.uniform(low=0, high=self._one_way_wall_radius)
            y = np.sqrt(
                np.random.uniform(low=0, high=(self._one_way_wall_radius**2 - x**2))
            )
            return np.array([x, y])

        def inverse_kinematics(pos):
            x, y = pos[0], pos[1]
            a1, a2 = self._actuators_length[0], self._actuators_length[1]
            d = (x**2 + y**2 - a1**2 - a2**2) / (2 * a1 * a2)
            if np.random.rand() < 0.5:
                theta2 = np.arccos(d)
                theta1 = np.arctan(y / x) - np.arctan(
                    (a2 * np.sin(theta2)) / (a1 + a2 * np.cos(theta2))
                )
            else:
                theta2 = -np.arccos(d)
                theta1 = np.arctan(y / x) - np.arctan(
                    (a2 * np.sin(theta2)) / (a1 + a2 * np.cos(theta2))
                )
            return theta1, theta2

        def forward_kinematics(theta1, theta2):
            a1, a2 = self._actuators_length[0], self._actuators_length[1]
            x = a1 * np.cos(theta1) + a2 * np.cos(theta1 + theta2)
            y = a1 * np.sin(theta1) + a2 * np.sin(theta1 + theta2)
            return np.array([x, y])

        target_1_pos = self.get_target_1_pos()
        while True:
            final_finger_pos = sample_xy() + target_1_pos
            if np.linalg.norm(final_finger_pos) > self._actuators_length.sum():
                continue
            if final_finger_pos[0] == 0.0:
                continue
            theta1, theta2 = inverse_kinematics(final_finger_pos)
            if np.isnan(theta1) or np.isnan(theta2):
                continue
            tmp_finger_pos = forward_kinematics(theta1, theta2)
            if np.linalg.norm(final_finger_pos - tmp_finger_pos) > 1e-5:
                theta1 += np.pi
            break
        return theta1, theta2

    def reset(self):
        if self._loca_mode == "eval":
            obs = super().reset()
            while self.check_inside_one_way_wall():
                obs = super().reset()
            return obs
        elif not self.is_phase_2():
            return super().reset()

        _ = self._env.reset()
        theta1, theta2 = self.sample_in_one_way_wall()
        with self._env._physics.reset_context():
            self._env._physics.named.data.qpos["shoulder"] = theta1
            self._env._physics.named.data.qpos["wrist"] = theta2
        obs = dict()
        obs["image"] = self.render().transpose(2, 0, 1).copy()
        return obs


class TimeLimit:
    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs, reward, done, info = self._env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            if "discount" not in info:
                info["discount"] = np.array(1.0).astype(np.float32)
            self._step = None
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self._env.reset()


class ActionRepeat:
    def __init__(self, env, amount):
        self._env = env
        self._amount = amount

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        done = False
        total_reward = 0
        current_step = 0
        while current_step < self._amount and not done:
            obs, reward, done, info = self._env.step(action)
            total_reward += reward
            current_step += 1
        return obs, total_reward, done, info


class NormalizeActions:
    def __init__(self, env):
        self._env = env
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low), np.isfinite(env.action_space.high)
        )
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        return gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self._env.step(original)


class ObsDict:
    def __init__(self, env, key="obs"):
        self._env = env
        self._key = key

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        spaces = {self._key: self._env.observation_space}
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        return self._env.action_space

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = {self._key: np.array(obs)}
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        obs = {self._key: np.array(obs)}
        return obs


class OneHotAction:
    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete)
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        shape = (self._env.action_space.n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.sample = self._sample_action
        return space

    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        if not np.allclose(reference, action):
            raise ValueError(f"Invalid one-hot action:\n{action}")
        return self._env.step(index)

    def reset(self):
        return self._env.reset()

    def _sample_action(self):
        actions = self._env.action_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


class RewardObs:
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        spaces = self._env.observation_space.spaces
        assert "reward" not in spaces
        spaces["reward"] = gym.spaces.Box(-np.inf, np.inf, dtype=np.float32)
        return gym.spaces.Dict(spaces)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs["reward"] = reward
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        obs["reward"] = 0.0
        return obs


class ResizeImage:
    def __init__(self, env, size=(64, 64)):
        self._env = env
        self._size = size
        self._keys = [
            k
            for k, v in env.obs_space.items()
            if len(v.shape) > 1 and v.shape[:2] != size
        ]
        print(f'Resizing keys {",".join(self._keys)} to {self._size}.')
        if self._keys:
            from PIL import Image

            self._Image = Image

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def obs_space(self):
        spaces = self._env.obs_space
        for key in self._keys:
            shape = self._size + spaces[key].shape[2:]
            spaces[key] = gym.spaces.Box(0, 255, shape, np.uint8)
        return spaces

    def step(self, action):
        obs = self._env.step(action)
        for key in self._keys:
            obs[key] = self._resize(obs[key])
        return obs

    def reset(self):
        obs = self._env.reset()
        for key in self._keys:
            obs[key] = self._resize(obs[key])
        return obs

    def _resize(self, image):
        image = self._Image.fromarray(image)
        image = image.resize(self._size, self._Image.NEAREST)
        image = np.array(image)
        return image


class RenderImage:
    def __init__(self, env, key="image"):
        self._env = env
        self._key = key
        self._shape = self._env.render().shape

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def obs_space(self):
        spaces = self._env.obs_space
        spaces[self._key] = gym.spaces.Box(0, 255, self._shape, np.uint8)
        return spaces

    def step(self, action):
        obs = self._env.step(action)
        obs[self._key] = self._env.render("rgb_array")
        return obs

    def reset(self):
        obs = self._env.reset()
        obs[self._key] = self._env.render("rgb_array")
        return obs