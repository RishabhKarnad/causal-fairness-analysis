from lime.lime_tabular import LimeTabularExplainer

def lime_explain(model, data, feature_names, num_samples = 100):
    sampled_data = data.sample(num_samples)
    if(len(feature_names) != len(data.columns)):
        raise ValueError("Number of feature names must match number of columns in data")
    data_array = sampled_data.values
    explainer = LimeTabularExplainer(data_array, feature_names=feature_names, class_names=['0', '1'])
    exp = [explainer.explain_instance(row, model.predict_proba) for row in data_array]
    return exp

