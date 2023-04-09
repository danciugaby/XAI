import matplotlib.pyplot as plt
import numpy as np


def plot_shap_values(shap_values, feature_names):
    # Convert shap values to a numpy array
    shap_values = np.array(shap_values)

    # Compute the mean absolute shap value for each feature
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    # Sort the features by their mean absolute shap value
    sorted_idx = np.argsort(mean_abs_shap)

    # Create a horizontal bar chart of the mean absolute shap values
    plt.barh(range(len(sorted_idx)), mean_abs_shap[sorted_idx])
    plt.yticks(range(len(sorted_idx)), feature_names[sorted_idx])
    plt.xlabel("Mean absolute SHAP value")
    plt.title("Feature importance")
    plt.show()