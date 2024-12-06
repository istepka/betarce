import os
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay

# from scikit_models import scikit_predict_proba, scikit_predict_proba_fn
from .robx import counterfactual_stability, counterfactual_stability_test


def __save_plot(plot: object, save_path: str, dpi: int = 300) -> None:
    '''
    Save a plot.
    
    Parameters:
        - plot: the plot (object)
        - save_path: the path to save the plot (str)
    '''
    
    # Make sure the directory exists    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the plot
    plot.savefig(save_path, dpi=dpi)

def plot_crisp_decision_boundary(
        modelMLP: object, 
        X: np.ndarray, 
        y: np.ndarray,
        save_path: str = None,
        show: bool = True,
        ) -> None:
    '''
    Plots the decision boundary of a classifier model along with the data points.
    
    Parameters:
        modelMLP (object): The trained classifier model.
        X (np.ndarray): The input feature matrix.
        y (np.ndarray): The target labels.
        save_path (str, optional): The file path to save the plot. Defaults to None.
        show (bool, optional): Whether to display the plot. Defaults to True.
    '''
    x_span = np.linspace(min(X[:,0]) - 0.25, max(X[:,0]) + 0.25)
    y_span = np.linspace(min(X[:,1]) - 0.25, max(X[:,1]) + 0.25)
    xx, yy = np.meshgrid(x_span, y_span)
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_,yy_]
    pred_func = modelMLP.predict(grid)
    z = pred_func.reshape(xx.shape)
    
    plt.contourf(xx,yy,z, cmap=plt.cm.coolwarm, alpha=0.5)
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.title("Decision boundary")
    plt.xlabel("x1")
    plt.ylabel("x2")
    
    if save_path:
        __save_plot(plt, save_path)
    
    if show:
        plt.show()

def plot_model_probabilites(model: object, 
                           X: np.ndarray, 
                           y: np.ndarray,
                           save_path: str = None,
                           show: bool = True,
                           ) -> None:
    '''
    Plot the probabilities of belonging to class 1 for a given model.
    
    Parameters:
        - model: the model (object)
        - X: the data (np.ndarray)
        - y: the labels (np.ndarray)
        - save_path: the path to save the plot (str)
        - show: whether to show the plot or not (bool)
    '''
    
    grid = np.linspace(np.min(X), np.max(X), 100)
    xx, yy = np.meshgrid(grid, grid)
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Predict probabilities for each point in the grid
    probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)
    
    # Plot the probabilities
    plt.contourf(xx, yy, probs, 25, cmap="RdBu_r", vmin=0, vmax=1)
    plt.title("Probability of belonging to class 1")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.colorbar()
    
    if save_path:
        __save_plot(plt, save_path)
    
    if show:
        plt.show()
    
def comprehensive_counterfactual_stability_plot(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    variances: list = [0.01, 0.1, 0.25],
    N: int = 100,
    tau: float = 0.5,
    grid_density_for_contourf: int = 50,
    save_path: str = None,
    show: bool = True,
    ) -> None:
    '''
    Plot the counterfactual stability for a given model.

    Parameters:
        - model: the model (object)
        - X: the data (np.ndarray)
        - y: the labels (np.ndarray)
        - save_path: the path to save the plot (str)
        - show: whether to show the plot or not (bool)
    '''
    
    model_name = model.__class__.__name__
    
    # Prepare the grid
    xx = np.linspace(np.min(X[:,0]), np.max(X[:,0]), grid_density_for_contourf)
    yy = np.linspace(np.min(X[:,1]), np.max(X[:,1]), grid_density_for_contourf)
    xx, yy = np.meshgrid(xx, yy)
    grid = np.c_[xx.ravel(), yy.ravel()]

    predict_fn = scikit_predict_proba_fn(model)
    
    def __run_in_parallel(variance, e):
        # Calculate the counterfactual stability for all points in the grid
        results_cf_stability = np.zeros(shape=(xx.shape[0], xx.shape[1]))
        results_cf_stability_test = np.zeros(shape=(xx.shape[0], xx.shape[1]))
        probabilities = model.predict_proba(grid)[:, 1].reshape(xx.shape)
    
        for i in tqdm(range(xx.shape[0]), desc=f'Calculating counterfactual stability (experiment {e+1}/{len(variances)})'):
            for j in range(xx.shape[1]):
                c = 1 if probabilities[i,j] > 0.5 else 0
                results_cf_stability[i,j] = counterfactual_stability(np.array([xx[i,j], yy[i,j]]), predict_fn, variance, N=N)
                
                # Test if the counterfactual stability is higher than the threshold tau
                results_cf_stability_test[i,j] = counterfactual_stability_test(results_cf_stability[i,j], tau=tau)

        fig, ax  = plt.subplots(figsize=(15,4), ncols=4, nrows=1)
        ax = ax.flatten()

        # Plot original data with discrete colorbar
        cmap = ListedColormap(['#4e48aa', '#e91911'])
        classes = [0, 0.5, 1]  # Set boundaries for two classes

        scatter = ax[0].scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, norm=plt.Normalize(min(classes), max(classes)))
        ax[0].set_title('Original data')
        ax[0].set_xlabel('x1')
        ax[0].set_ylabel('x2')
        cbar0 = plt.colorbar(scatter, ax=ax[0], boundaries=classes, ticks=[0, 0.5, 1])
        cbar0.set_label('Class', rotation=270, fontsize=9, labelpad=15)


        contour = ax[1].contourf(xx, yy, results_cf_stability, 25, cmap="RdBu_r", vmin=0, vmax=1)
        ax[1].set_title(f'Counterfactual stability')
        ax[1].set_xlabel('x1')
        ax[1].set_ylabel('x2')

        cbar1 = plt.colorbar(contour, ax=ax[1])
        cbar1.set_label('Counterfactual Stability', rotation=270, fontsize=9, labelpad=15)
        
        cmap = ListedColormap(['#cc0000', '#33cc33'])
        contour = ax[2].contourf(xx, yy, results_cf_stability_test, 25, cmap=cmap, vmin=0, vmax=1)
        ax[2].set_title(f'Counterfactual stability test')
        ax[2].set_xlabel('x1')
        ax[2].set_ylabel('x2')
        
        boundaries = [0, tau, 1]
        cvar2 = plt.colorbar(contour, ax=ax[2], boundaries=boundaries, ticks=[0, 1])
        cvar2.set_label('Counterfactual Stability Test', rotation=270, fontsize=9, labelpad=15)
        cvar2.set_ticklabels(['fail', 'pass'])
        
        
        contour = ax[3].contourf(xx, yy, probabilities, 25, cmap="RdBu_r", vmin=0, vmax=1)
        ax[3].set_title(f'Probabilities (of belonging to class 1)')
        ax[3].set_xlabel('x1')
        ax[3].set_ylabel('x2')
        cbar4 = plt.colorbar(contour, ax=ax[3])
        cbar4.set_label('Probability', rotation=270, fontsize=9, labelpad=15) 

        plt.suptitle(f'Counterfactual stability (Normal distribution identity variance = {variance} and N={N} samples) for {model_name}')
        plt.tight_layout(w_pad=2)
        
        if save_path:
            if '.' not in save_path:
                __save_plot(plt, save_path)
            else:
                new_save_path = save_path.replace('.', f'_T{tau}_V{variance}_N{N}.')
                __save_plot(plt, new_save_path)
                
        if show:
            plt.show()
    
    
    processes = []
    
    for e, variance in enumerate(variances):
        p = mp.Process(target=__run_in_parallel, args=(variance, e))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
             
def plot_calibration_curves(
            clf_list: list,
            X: np.ndarray,
            y: np.ndarray,
            save_path: str = None,
            show: bool = True,
            ) -> None:
    '''
    Plot the calibration curves for a list of classifiers.
    
    Parameters:
        - clf_list: the list of classifiers (list of tuples (clf, name))
        - X: the data (np.ndarray)
        - y: the labels (np.ndarray)
        - save_path: the path to save the plot (str)
        - show: whether to show the plot or not (bool)
    '''

    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(2, 3)
    colors = plt.get_cmap("Dark2")

    ax_calibration_curve = fig.add_subplot(gs[:1, :3])
    calibration_displays = {}
    for i, (clf, name) in enumerate(clf_list):
        display = CalibrationDisplay.from_estimator(
            clf,
            X,
            y,
            n_bins=10,
            name=name,
            ax=ax_calibration_curve,
            color=colors(i),
        )
        calibration_displays[name] = display

    ax_calibration_curve.grid()

    # Add histogram
    grid_positions = [(1, 0), (1, 1), (1, 2)]
    for i, (_, name) in enumerate(clf_list):
        row, col = grid_positions[i]
        ax = fig.add_subplot(gs[row, col])

        ax.hist(
            calibration_displays[name].y_prob,
            range=(0, 1),
            bins=10,
            label=name,
            color=colors(i),
        )
        ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

    plt.tight_layout()
    
    if save_path:
        __save_plot(plt, save_path)
        
    if show:
        plt.show()

def plot_decision_boundary_with_artifacts(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    artifacts: list[dict],
    save_path: str = None,
    show: bool = True,
    ) -> None:
    '''
    Plot the decision boundary of a model along with the artifacts.
    
    Parameters:
        - model: the model (object)
        - X: the data (np.ndarray)
        - y: the labels (np.ndarray)
        - artifacts: list of artifacts (list[dict]) - [{'coords': np.ndarray, 'color': str, 'label': str, 'marker': str}]
        - save_path: the path to save the plot (str)
        - show: whether to show the plot or not (bool)
    '''

    # Prepare the grid
    xx = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100)
    yy = np.linspace(np.min(X[:,1]), np.max(X[:,1]), 100)
    xx, yy = np.meshgrid(xx, yy)
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    

    fig, ax  = plt.subplots(figsize=(5,5))
    
    # Plot the crisp decision boundary
    preds = model.predict(grid).reshape(xx.shape)
    
    ax.contourf(xx, yy, preds, 25, cmap="RdBu_r", vmin=0, vmax=1)
    ax.set_title("Decision boundary")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    
    
    
    # Plot the artifacts
    # [{'coords': np.ndarray, 'color': str, 'label': str, 'marker': str}]
    for artifact in artifacts:
        ax.scatter(artifact['coords'][0], artifact['coords'][1], c=artifact['color'], label=artifact['label'], marker=artifact['marker'])
        
    plt.legend()
    
    if save_path:
        __save_plot(plt, save_path)
        
    if show:
        plt.show()
 
 
def plot_decision_boundary_with_artifacts(
    predict_crisp_fn: callable,
    X: np.ndarray,
    y: np.ndarray,
    artifacts: list[dict],
    save_path: str = None,
    show: bool = True,
    ) -> None:
    '''
    Plot the decision boundary of a model along with the artifacts.
    
    Parameters:
        - model: the model (object)
        - X: the data (np.ndarray)
        - y: the labels (np.ndarray)
        - artifacts: list of artifacts (list[dict]) - [{'coords': np.ndarray, 'color': str, 'label': str, 'marker': str}]
        - save_path: the path to save the plot (str)
        - show: whether to show the plot or not (bool)
    '''

    # Prepare the grid
    xx = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100)
    yy = np.linspace(np.min(X[:,1]), np.max(X[:,1]), 100)
    xx, yy = np.meshgrid(xx, yy)
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    

    fig, ax  = plt.subplots(figsize=(5,5))
    
    # Plot the crisp decision boundary
    preds = predict_crisp_fn(grid).reshape(xx.shape)
    
    ax.contourf(xx, yy, preds, 25, cmap="RdBu_r", vmin=0, vmax=1)
    ax.set_title("Decision boundary")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    
    
    
    # Plot the artifacts
    # [{'coords': np.ndarray, 'color': str, 'label': str, 'marker': str}]
    for artifact in artifacts:
        ax.scatter(artifact['coords'][0], artifact['coords'][1], c=artifact['color'], label=artifact['label'], marker=artifact['marker'])
        
    plt.legend()
    
    if save_path:
        __save_plot(plt, save_path)
        
    if show:
        plt.show()

def plot_crisp_decision_boundary(
        predict_crisp_fn: callable, 
        X: np.ndarray, 
        y: np.ndarray,
        save_path: str = None,
        show: bool = True,
        ) -> None:
    '''
    Plots the decision boundary of a classifier model along with the data points.
    
    Parameters:
        predict_crisp_fn: the predict function of the model (callable)
        X (np.ndarray): The input feature matrix.
        y (np.ndarray): The target labels.
        save_path (str, optional): The file path to save the plot. Defaults to None.
        show (bool, optional): Whether to display the plot. Defaults to True.
    '''
    x_span = np.linspace(min(X[:,0]) - 0.25, max(X[:,0]) + 0.25)
    y_span = np.linspace(min(X[:,1]) - 0.25, max(X[:,1]) + 0.25)
    xx, yy = np.meshgrid(x_span, y_span)
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_,yy_]
    pred = predict_crisp_fn(grid)
    z = pred.reshape(xx.shape)
    
    plt.contourf(xx,yy,z, cmap=plt.cm.coolwarm, alpha=0.5)
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.title("Decision boundary")
    plt.xlabel("x1")
    plt.ylabel("x2")
    
    if save_path:
        __save_plot(plt, save_path)
    
    if show:
        plt.show()