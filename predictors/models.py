from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pandas as pd
from joblib import dump, load
import os
import numpy as np

def model(train_df_path, test_df_path, model_type):
    if(model_type=='LR'):
        clf = LogisticRegression()
    if(model_type=='mlp'):
        clf = MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(5, 2), random_state=1, max_iter=1000)
    if(model_type=='dt'):
        clf = DecisionTreeClassifier(random_state=0)
    
    train_df = pd.read_csv(train_df_path)
    train_df.columns = ['z', 'x', 'w', 'y']
    
    train_df['z'] = train_df['z'].astype(float)
    train_df['x'] = train_df['x'].astype(float)
    train_df['w'] = train_df['w'].astype(float)
    # train_df['z'] = (train_df['z'] - train_df['z'].mean()) / train_df['z'].std()
    # train_df['x'] = (train_df['x'] - train_df['x'].mean()) / train_df['x'].std()
    # train_df['w'] = (train_df['w'] - train_df['w'].mean()) / train_df['w'].std()
    train_df['y'] = train_df['y'].astype(int)
    
    test_df = pd.read_csv(test_df_path)
    test_df.columns = ['z', 'x', 'w', 'y']
    test_df['y'] = test_df['y'].astype(int)
    test_df['z'] = test_df['z'].astype(float)
    test_df['x'] = test_df['x'].astype(float)
    test_df['w'] = test_df['w'].astype(float)
    # test_df['z'] = (test_df['z'] - train_df['z'].mean()) / train_df['z'].std()
    # test_df['x'] = (test_df['x'] - train_df['x'].mean()) / train_df['x'].std()
    # test_df['w'] = (test_df['w'] - train_df['w'].mean()) / train_df['w'].std()
    
    scm_type = train_df_path.split('/')[2].split('_')[0]
    data_type = train_df_path.split('/')[2].split('_')[1]
    
    x = train_df.drop(columns=['y'])
    y = train_df['y']
    
    clf.fit(x, y)
    
    test_x = test_df.drop(columns=['y'])
    test_y = test_df['y']
    predictions = clf.predict(test_x)
    train_predictions = clf.predict(x)
    # print no of zeros and ones in predictions
    #print(np.sum(predictions==0), np.sum(predictions==1))
    #print(test_df['y'].value_counts())
    #print(accuracy_score(test_df['y'], predictions))
    # if os path not exisits scm type create it
    if(os.path.exists(f"./trained_models/{scm_type}/{data_type}" == False)):
        os.makedirs(f"./trained_models/{scm_type}/{data_type}", exist_ok=True)
    dump(clf, f'./trained_models/{scm_type}/{data_type}/{model_type}.joblib')
    return clf, accuracy_score(test_y, predictions), accuracy_score(y, train_predictions), f1_score(test_y, predictions), f1_score(y, train_predictions)


