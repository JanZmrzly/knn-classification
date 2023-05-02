import pandas as pd

import sys
import joblib
import datetime

def dummies(df:pd.DataFrame):
    non_numeriacal = ["Gender", "Ever_Married", "Graduated",
                    "Profession", "Spending_Score", "Var_1"]
    x = pd.get_dummies(df, columns=non_numeriacal)
    return x

def main():
    DATA_SET = "data\\Train.csv"
    MODEL = "models\\20230422_2035_kkn_model.joblib"

    if len(sys.argv) >= 2:
        DATA_SET = sys.argv[1]
    elif len(sys.argv) >= 3:
        MODEL = sys.argv[2]
    else:
        raise Exception("Few input arguments were given. Two are required (Dataset and Model)")

    model = joblib.load(MODEL)

    df = pd.read_csv(DATA_SET)
    df = df.fillna(df.median(numeric_only=True).round())
    df = df.dropna(how="any")
    df = df.drop("ID", axis=1)

    if "Segmentation" in df.columns:
        df["Segmentation"] = df["Segmentation"].replace({'A': 0, 'B': 1, 'C': 2, 'D': 3})
        x = dummies(df)
        y = x.pop("Segmentation")
        result = model.score(x, y)
        print(f"[INFO]\tAccuracy: {round(df.shape[0] * result)}/{df.shape[0]}")
    else:
        x = dummies(df)
        result = model.predict(x)
        df["Prediction"] = result
        df["Prediction"] = df["Prediction"].replace({0: 'A', 1: 'B', 2: 'C', 3: 'D'})

        now = datetime.datetime.now()
        sv = now.strftime("%Y%m%d_%H%M") 
        df.to_csv(f"results/{sv}_prediction")
        print(f"[INFO]\tThe results were saved in a folder -> .results\\{sv}_prediction.csv")

if __name__ == "__main__":
    main()