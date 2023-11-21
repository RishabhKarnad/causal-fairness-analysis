import shap

def shap_explain(model, data, num_samples = 100):
    data_array = data.values
    sampled_values = shap.sample(data_array, num_samples)
    explainer = shap.KernelExplainer(model.predict_proba, sampled_values)
    shap_values = explainer.shap_values(sampled_values)
    return shap_values

