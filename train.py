import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.neighbors import NeighborhoodComponentsAnalysis, KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.neural_network import MLPClassifier

import sys
import joblib
import datetime
import threading
import time

def plot_data(df:pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(20, 4))
    sns.scatterplot(data=df, ax=axes[0], x="Age", y="Work_Experience", hue="Segmentation", palette="bright")
    sns.scatterplot(data=df, ax=axes[1], x="Age", y="Family_Size", hue="Segmentation", palette="bright")
    sns.scatterplot(data=df, ax=axes[2], x="Age", y="Profession", hue="Segmentation", palette="bright")
    plt.show()

def plot_boxes(df:pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    sns.boxplot(ax=axes[0], x=df["Age"], palette=["m"])
    sns.boxplot(ax=axes[1], x=df["Work_Experience"], palette=["g"])
    sns.boxplot(ax=axes[2], x=df["Family_Size"], palette=["b"])
    plt.show()

def remove_outliers(df:pd.DataFrame, column:str, z_score:int):
    z = abs(stats.zscore(df[column]))
    df = df[(z < z_score)]
    return df

def main():
    TRAIN_SET = "data/Train.csv"
    RUNNING = True

    def thread():
        while RUNNING:
            print("[INFO]\tLoading...")
            time.sleep(0.5)

    mthread = threading.Thread(target=thread)
    mthread.start()

    if len(sys.argv) > 1:
        TRAIN_SET = sys.argv[1]

    df = pd.read_csv(TRAIN_SET)
    df = df.fillna(df.median(numeric_only=True).round())
    df = df.dropna(how="any")
    df = df.drop("ID", axis=1)

    # plot_data(df)
    # plot_boxes(df)

    df = remove_outliers(df, "Age", 2)
    df = remove_outliers(df, "Work_Experience", 3)
    df = remove_outliers(df, "Family_Size", 3)

    # plot_boxes(df)

    df["Segmentation"] = df["Segmentation"].replace({'A': 0, 'B': 1, 'C': 2, 'D': 3})
    y = df.pop("Segmentation")

    non_numeriacal = ["Gender", "Ever_Married", "Graduated",
                    "Profession", "Spending_Score", "Var_1"]
    x = pd.get_dummies(df, columns=non_numeriacal)

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        stratify=y, 
                                                        test_size=0.8, 
                                                        random_state=42)

    nca = NeighborhoodComponentsAnalysis(random_state=42)
    knn = KNeighborsClassifier(n_neighbors=4)
    nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
    nca_pipe.fit(x_train, y_train)

    now = datetime.datetime.now()
    sv = now.strftime("%Y%m%d_%H%M") 
    joblib.dump(nca_pipe, f"models/{sv}_kkn_model.joblib")

    clf = svm.SVC()
    clf.fit(x, y)

    nn = MLPClassifier(solver='adam', alpha=1e-6,
                    hidden_layer_sizes=[512, 128], random_state=1)
    nn.fit(x_train, y_train)

    RUNNING = False
    mthread.join()

    print("")
    print("[RESULTS]")
    print(f"KNN Accurancy:\t{nca_pipe.score(x_test, y_test):.3f}")
    print(f"SVM Accurancy:\t{clf.score(x_test, y_test):.3f}")
    print(f"NN Accurancy:\t{nn.score(x_test, y_test):.3f}")

if __name__ == "__main__":
    main()