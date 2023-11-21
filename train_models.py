from predictors.models import model
from prettytable import PrettyTable
import os
from glob import glob




def main():
    model_list = ["LR", "mlp", "dt"]
    x = PrettyTable()
    x.field_names = ["Model", "SCM", "Data", "Accuracy", "F1 Score",  "Training Accuracy", "Training F1 Score"]
    f = glob("./data/*_train.csv")
    for train_file in f:
        test_file = train_file.replace('train', 'test')
        #print(train_file, test_file)
        scm = train_file.split('/')[2].split('_')[0]
        data = train_file.split('/')[2].split('_')[1]
        for model_type in model_list:
            clf, accuracy, training_accuracy,  f1_score, training_f1_score = model(train_file, test_file, model_type)
            x.add_row([model_type, scm, data, round(accuracy,5), round(f1_score,5), round(training_accuracy,5), round(training_f1_score,5)])
    print(x)
    
if __name__ == "__main__":
    main()