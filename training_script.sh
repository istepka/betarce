EXPERIMENT_NAME="robust"
CONFIG_FILENAME="config_exp"

# PATHS
BASE_PATH="/home/inf148179/robust-cf/kdd/$EXPERIMENT_NAME/"
MODEL_PATH="/home/inf148179/robust-cf/kdd/$EXPERIMENT_NAME/models/"
LOG_PATH="/home/inf148179/robust-cf/kdd/$EXPERIMENT_NAME/logs/"
RESULT_PATH="/home/inf148179/robust-cf/kdd/$EXPERIMENT_NAME/results/"

# SWEEP
robust_method=betarob,robx
datasets=[car_eval],[rice]
ex_type=[Architecture],[Bootstrap],[Seed]
base_cf=[gs],[dice],[face]
model_type_to_use=neural_network


python src/experimentsv3.py -cn $CONFIG_FILENAME --multirun \
    experiments_setup.robust_cf_method=$robust_method \
    experiments_setup.datasets=$datasets \
    experiments_setup.ex_types=$ex_type \
    experiments_setup.base_counterfactual_method=$base_cf \
    experiments_setup.model_type_to_use=$model_type_to_use \
    general.result_path=$RESULT_PATH \
    general.log_path=$LOG_PATH \
    general.model_path=$MODEL_PATH \