import sys, pytest, random
import numpy as np

sys.path.append('../pydynet')

np.random.seed(0)
random.seed(0)

type_list = [np.float16, np.float32, np.float64]
