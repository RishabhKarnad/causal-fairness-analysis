import os
from utils import *
from metrics.lime import lime_explain
from metrics.shap import shap_explain
import warnings
import numpy as np
from tqdm import tqdm
warnings.filterwarnings("ignore")
from prettytable import PrettyTable

data_path ="./data"
models_path = "./trained_models"
x = PrettyTable()
x.field_names = ["Model", "SCM", "Data", "Explainability Method",'z','x','w']

for scm_type in ['fair','unfair']:
    for data_type in ['linear','nonlinear']:
        test_data_path = f"./data/{scm_type}_{data_type}_test.csv"
        test_data = load_test_data(test_data_path)
        test_data.columns = ['z', 'x', 'w', 'y']
        features = test_data.drop(columns=['y'])
        feature_names = ['z', 'x', 'w']
        #print(features.columns)
        for model_name in ["LR",'mlp','dt']:#os.listdir(os.path.join(models_path,scm_type,data_type)):
            model_path = os.path.join(models_path,scm_type,data_type,model_name + ".joblib")
            model = load_model(model_path)
            print(scm_type, data_type, model_name.split(".")[0])
            #LIME
            lime_explaination = lime_explain(model, features, feature_names)
            max_feature_contribution_lime = {}
            for explaination in lime_explaination:
                for f_threshold, value in explaination.as_list():
                    if('z' in f_threshold):
                        feature = 'z'
                    elif('x' in f_threshold):
                        feature = 'x'
                    elif('w' in f_threshold):
                        feature = 'w'
                    if feature in max_feature_contribution_lime:
                        if abs(value) > max_feature_contribution_lime[feature]:
                            max_feature_contribution_lime[feature] = abs(value)
                    else:
                        max_feature_contribution_lime[feature] = abs(value)
            # # shap
            
            
            #print(max_feature_contribution_lime.keys())
            shap_values = shap_explain(model, features)
            max_feature_contribution_shap = {}
            max_shap = np.max(np.abs(shap_values), axis=1)
            #print(max_shap)
            for idx, feature in enumerate(feature_names):
                max_feature_contribution_shap[feature] = max_shap[0][idx]
            # print(shap_values.shape)
            
            print(max_feature_contribution_shap.keys())
            
            x.add_row([model_name.split(".")[0], scm_type, data_type, 'LIME', max_feature_contribution_lime['z'], max_feature_contribution_lime['x'], max_feature_contribution_lime['w']])
            x.add_row([model_name.split(".")[0], scm_type, data_type, 'SHAP', max_feature_contribution_shap['z'], max_feature_contribution_shap['x'], max_feature_contribution_shap['w']])            
print(x)