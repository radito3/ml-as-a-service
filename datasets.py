from sklearn import datasets


def get_iris_dataset():
    iris_x, iris_y = datasets.load_iris(return_X_y=True)
    return {"x": iris_x, "y": iris_y}


def get_wine_dataset():
    wine_x, wine_y = datasets.load_wine(return_X_y=True)
    return {"x": wine_x, "y": wine_y}


def generate_random_sample():
    n_samples = 300
    n_features = 3
    n_classes = 5
    x, y = datasets.make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_clusters_per_class=1,
        n_classes=n_classes,
        random_state=42
    )
    return {"x": x, "y": y}
