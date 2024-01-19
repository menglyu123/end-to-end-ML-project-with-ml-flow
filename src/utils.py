import os,sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill


def save_object(obj, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as fp:
            dill.dump(obj, fp)

    except Exception as e:
        raise CustomException(e, sys)
        