import joblib
import pandas as pd


def load_test_data(path):
    return pd.read_csv(path)

def load_model(path):
    return joblib.load(path)

