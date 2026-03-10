import xgboost
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import random
import numpy as np
from tqdm import tqdm
import os
import sys
import pandas as pd
import requests
import torch
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import apscheduler
import pymongo

print("Torch: ", torch.__version__)
print("XGBoost: ", xgboost.__version__)
print("Pandas: ", pd.__version__)
print("NumPy: ", np.__version__)
print("Joblib: ", joblib.__version__)
print("Pickle: ", pickle.format_version)
print("Seaborn: ", sns.__version__)
print("Apscheduler: ", apscheduler.__version__)
print("Pymongo: ", pymongo.__version__)