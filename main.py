import time

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
)
from sklearn.svm import SVC

from src.svm import SVMClassifier


def generate_sparse_2D_dataset(
    n_samples: int, file_name: str = "datasets/two-dim-dataset-for-visualisation.csv"
):
    samples_per_class = n_samples // 2  # Controls sparsity
    class0_center = [1, 1]
    class1_center = [5, 5]
    noise_level = 1  # Keep low for clear separation

    # Generate data
    class0 = np.random.normal(
        loc=class0_center, scale=noise_level, size=(samples_per_class, 2)
    )
    class1 = np.random.normal(
        loc=class1_center, scale=noise_level, size=(samples_per_class, 2)
    )

    # Create DataFrame
    df = pd.DataFrame(np.vstack([class0, class1]), columns=["X1", "X2"])
    df["Outcome"] = [0] * samples_per_class + [1] * samples_per_class

    # Save to CSV
    df.to_csv(file_name, index=False)


file_path = "datasets/two-dim-dataset-for-visualisation.csv"
generate_sparse_2D_dataset(1000, file_path)
df = pd.read_csv(file_path)

X = df.drop(
    [
        "Outcome",
    ],
    axis=1,
)
y = df["Outcome"]
y = y.replace(0, -1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

scaler = StandardScaler()

scaled_X_train = np.array(scaler.fit_transform(X_train))
scaled_X_test = np.array(scaler.transform(X_test))

y_test = np.array(y_test)


def plot_svm_linear_boundary(clf, X, y):
    plt.figure(figsize=(8, 6))

    # Plot data points
    plt.scatter(
        X[y == -1][:, 0], X[y == -1][:, 1], color="blue", label="Class -1", s=25
    )
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color="red", label="Class 1", s=25)

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)

    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    ax.contour(
        XX,
        YY,
        Z,
        colors="k",
        levels=[-2, -1, 0, 1, 2],
        alpha=0.5,
        linestyles=["--", "--", "-", "--", "--"],
    )

    plt.title("SVM Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()

    plt.show()


### SVC made by us
svc_model = SVMClassifier(C=10, kernel="rbf", tol=1e-3, max_iter=100000)


start_time = time.perf_counter_ns()
svc_model.fit(scaled_X_train, np.array(y_train))
print(f"Process finished in {time.perf_counter_ns() - start_time}ns")

svc_pred = svc_model.predict(np.array(scaled_X_test))
plot_svm_linear_boundary(svc_model, scaled_X_train, np.array(y_train))

print(classification_report(y_test, svc_pred))


### SVC From sklearn
svc_model = SVC(kernel="linear")
svc_model.fit(scaled_X_train, y_train)

svc_pred = svc_model.predict(scaled_X_test)
# plot_svm_linear_boundary(svc_model, scaled_X_train, np.array(y_train))


print(classification_report(y_test, svc_pred))
