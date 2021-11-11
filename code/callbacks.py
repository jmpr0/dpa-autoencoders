import csv
import os
from time import process_time as time

import numpy as np
import pandas as pd
import weightwatcher as ww
from tensorflow.keras.callbacks import *


class TimeEpochs(Callback):
    """
	Callback used to calculate per-epoch time
	"""

    def on_train_begin(self, logs):
        logs = logs or {}
        return

    def on_epoch_begin(self, batch, logs):
        logs = logs or {}
        self.epoch_time_start = time()

    def on_epoch_end(self, batch, logs):
        logs = logs or {}
        logs['time'] = (time() - self.epoch_time_start)
