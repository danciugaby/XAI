from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors


class LocalSurrogateModels:

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.nearest_neighbors = NearestNeighbors(n_neighbors=5)
        self.nearest_neighbors.fit(data)

    def explain_instance(self, x):
        neighbors = self.nearest_neighbors.kneighbors([x])[1][0]
        local_data = self.data[neighbors]
        local_target = self.model.predict(local_data)
        surrogate_model = LinearRegression()
        surrogate_model.fit(local_data, local_target)
        return surrogate_model