import shap

def compute_shapley_values(model, X):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    return shap_values.values