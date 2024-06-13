import tempfile
import subprocess
from os import remove

import numpy as np


class TestPostprocess:
    def setup_method(self):
        scores = np.zeros((50, 30, 40))
        scores[15, 25, 30] = 30
        scores[15, 25, 30] = 30
