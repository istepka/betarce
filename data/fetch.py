from ucimlrepo import fetch_ucirepo

# fetch dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

# data (as pandas dataframes)
X = breast_cancer_wisconsin_diagnostic.data.features.copy()
y = breast_cancer_wisconsin_diagnostic.data.targets.copy()

print(X.shape, y.shape)

final = X.copy()
final["diagnosis"] = y.to_numpy().flatten()

final.to_csv("data/breast_cancer.csv", index=False)


# fetch dataset
wine_quality = fetch_ucirepo(id=186)

# data (as pandas dataframes)
X = wine_quality.data.features.copy()
y = wine_quality.data.targets.copy()

final = X.copy()
binary = y.to_numpy().flatten() > 5
final["quality"] = binary

final.to_csv("data/wine_quality.csv", index=False)


#
car_eval = fetch_ucirepo(id=19)
X = car_eval.data.features.copy()
y = car_eval.data.targets.copy()

final = X.copy()
final["acceptability"] = y.to_numpy().flatten()

final.to_csv("data/car_eval.csv", index=False)

# rice
rice = fetch_ucirepo(id=545)
X = rice.data.features.copy()
y = rice.data.targets.copy()

final = X.copy()
final["class"] = y.to_numpy().flatten()

final.to_csv("data/rice.csv", index=False)
