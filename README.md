# CodeAlpha_irisFlowerClassification
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

# 1. Load the dataset ---------------------------------------------------
def load_data():
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "saurabh00007/iriscsv",
            "Iris.csv",            
        )
    except Exception as e:
        print(f"Kaggle load failed ({e}); using scikit-learn's built-in Iris data.")
        from sklearn.datasets import load_iris
        iris = load_iris(as_frame=True)
        df = iris.frame.copy()
        df["Species"] = iris.target_names[iris.target]
        df = df.drop(columns="target")
        df.columns = ["SepalLengthCm", "SepalWidthCm",
                      "PetalLengthCm", "PetalWidthCm", "Species"]
    return df

df = load_data()
print("First 5 records:\n", df.head())
df = df.drop(columns=[c for c in ["Id"] if c in df.columns])

# 2. Explore ------------------------------------------------------------
print("\nShape:", df.shape)
print("Missing values:\n", df.isnull().sum())
print("Class balance:\n", df["Species"].value_counts())
sns.pairplot(df, hue="Species")
plt.savefig("iris_pairplot.png", bbox_inches="tight")
plt.close()

# 3. Split --------------------------------------------------------------
X = df.drop(columns="Species")
y = df["Species"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Compare models with 5-fold cross-validation ------------------------
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=200)),
    "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"{name:20s} CV accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

# 5. Train final model and evaluate on the test set ---------------------
best_model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("\nTest accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
ConfusionMatrixDisplay(cm, display_labels=best_model.classes_).plot(cmap="Blues")
plt.title("Confusion Matrix - Test Set")
plt.savefig("iris_confusion_matrix.png", bbox_inches="tight")
plt.close()

# 6. Predict a new flower -----------------------------------------------
sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=X.columns)
print("Prediction for [5.1, 3.5, 1.4, 0.2]:", best_model.predict(sample)[0])
