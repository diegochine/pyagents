import gym
import numpy as np
from PIL import Image


class ObservationWrapper(gym.ObservationWrapper):
    """
    ViZDoom environments return dictionaries as observations, containing
    the main image as well other info.
    The image is also too large for normal training.
    This wrapper replaces the dictionary observation space with a simple
    Box space (i.e., only the RGB image), and also resizes the image to a
    smaller size.
    NOTE: Ideally, you should set the image size to smaller in the scenario files
          for faster running of ViZDoom. This can really impact performance,
          and this code is pretty slow because of this!
    """
    IMAGE_SHAPE = (60, 80)

    def __init__(self, env, shape=IMAGE_SHAPE):
        super().__init__(env)
        self.image_shape = shape
        self.image_shape_reverse = shape[::-1]

        # Create new observation space with the new shape
        num_channels = env.observation_space["rgb"].shape[-1]
        self._new_shape = (shape[0], shape[1], num_channels)
        self.observation_space = gym.spaces.Box(0., 1., shape=self._new_shape, dtype=np.float32)

    def observation(self, observation):
        img = Image.fromarray(observation["rgb"]).resize(self.image_shape_reverse)  # .convert('L')  # greyscale
        return np.array(img) / 255.
