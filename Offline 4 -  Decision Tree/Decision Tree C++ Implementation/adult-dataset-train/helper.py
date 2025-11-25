# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
# import numpy as np

# data = pd.read_csv('Iris.csv')

# X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
# y = data['Species']

# for i in range(1, 6):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     dt_classifier = DecisionTreeClassifier(max_depth=i, random_state=42, criterion='entropy')
#     dt_classifier.fit(X_train, y_train)
#     y_pred = dt_classifier.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred) * 100
#     print(f"Decision Tree Depth: {i}")
#     print(f"Decision Tree Accuracy: {accuracy:.2f} %\n")
    


import pandas as pd


columns = [
    "workclass", "workclass_code", "education", "education_num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

df = pd.read_csv("adult_imputed.data", header=None, names=columns, skipinitialspace=True)


attributes_types = [
    ("workclass", "categorical"),
    ("workclass_code", "numerical"),
    ("education", "categorical"),
    ("education_num", "numerical"),
    ("marital-status", "categorical"),
    ("occupation", "categorical"),
    ("relationship", "categorical"),
    ("race", "categorical"),
    ("sex", "categorical"),
    ("capital-gain", "numerical"),
    ("capital-loss", "numerical"),
    ("hours-per-week", "numerical"),
    ("native-country", "categorical"),
    ("income", "categorical"),
]

cat_cols = {col for col, typ in attributes_types if typ == "categorical"}

with open("c++help.txt", "w", encoding="utf-8") as f:
    f.write("vector<Attributes> attributes = {\n")
    for col, typ in attributes_types:
        if typ == "categorical":
            uniques = sorted(df[col].dropna().unique())
            val_str = ", ".join([f'"{str(v)}"' for v in uniques])
            f.write(f'    Attributes("{col}", "{typ}", {{{val_str}}}),\n')
        else:
            f.write(f'    Attributes("{col}", "{typ}", {{}}),\n')
    f.write("};\n")


