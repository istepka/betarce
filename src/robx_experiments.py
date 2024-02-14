from robx import *
from plot_helpers import *
from scikit_models import *
from create_data_examples import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay

np.random.seed(0)

X_train, y_train = create_two_donuts(random_state=0)
X_cal, y_cal = create_two_donuts(random_state=1)
X_test, y_test = create_two_donuts(random_state=2)

for model_name in ['mlp', 'rf']:
    model = load_model(f'models/{model_name}.joblib')

    model_isotonic = CalibratedClassifierCV(model, cv='prefit', method="isotonic", n_jobs=-1)
    model_isotonic.fit(X_cal, y_cal)

    model_sigmoid = CalibratedClassifierCV(model, cv='prefit', method="sigmoid", n_jobs=-1)
    model_sigmoid.fit(X_cal, y_cal)

    clf_list = [
        (model, f"{model_name}_Uncalibrated"),
        (model_isotonic, f"{model_name}_Isotonic"),
        (model_sigmoid, f"{model_name}_Sigmoid"),
    ]

    plot_calibration_curves(
        clf_list=clf_list,
        X=X_test,
        y=y_test,
        save_path=f'images/calibration/{model_name}_calibration_curves.png',
        show=False,
    )

    for i, (m, name) in enumerate(clf_list):
        print(f'Analysing counterfactual stability for {name}')
        
        comprehensive_counterfactual_stability_plot(
            model=m,
            X=X_train,
            y=y_train,
            variances=[0.01, 0.05, 0.1, 0.25],
            N=100,
            tau=0.5,
            grid_density_for_contourf=50,
            save_path=f'images/calibration/{name}/{name}_cf_stability.png',
            show=False,
        )

