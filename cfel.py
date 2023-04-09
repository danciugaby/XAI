import matplotlib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

class CFEL:

    def __init__(self, data, model):
        self.data = data
        self.model = model


    def predict(self, x):
        return self.model.predict(x)

    def find_counterfactuals(self, x, y, n=5):
        # Get feature importances
        importances = self.model.feature_importances_

        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Get the k most important features
        k = min(n, len(x))
        top_k_indices = indices[:k]

        # Generate counterfactual explanations
        counterfactuals = []
        j=0
        for idx in range(self.data.shape[0]):
            # Only consider examples with different labels
            if self.predict(self.data.values[idx].reshape(1, -1)) != y.index[j]:
                counterfactual = x.copy()
                # Set the top-k most important features to the corresponding feature in the candidate counterfactual
                for i in top_k_indices:
                    counterfactual[i] = self.data.values[idx][i]
                counterfactuals.append(counterfactual)
            j=j+1
        return counterfactuals

    # Generate counterfactual samples
    def generate_counterfactual_samples(x, perturbation_direction):
        epsilon = 0.1
        n_features = len(x)
        x_cf = np.tile(x, (n_features, 1))  # Create a copy of x for each feature
        for i in range(n_features):
            if perturbation_direction[i] == 'increase':
                x_cf[i, i] += epsilon
            elif perturbation_direction[i] == 'decrease':
                x_cf[i, i] -= epsilon
        return x_cf

    def plot_counterfactual(original_data, counterfactual_data):
        # plot the dataframes
        for idx in  range(len(counterfactual_data)):
            df = pd.DataFrame(counterfactual_data[idx])
            df.plot()
            plt.show()


