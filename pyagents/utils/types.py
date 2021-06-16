from typing import Callable, Optional, Union

import numpy as np
import tensorflow as tf

Int = Union[int, np.int16, np.int32, np.int64, tf.Tensor]
Float = Union[float, np.float16, np.float32, np.float64, tf.Tensor]
