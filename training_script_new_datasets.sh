EXPERIMENT_NAME="datasets"
CONFIG_FILENAME="config_exp"

# PATHS
BASE_PATH="/home/inf148179/robust-cf/kdd/$EXPERIMENT_NAME/"
MODEL_PATH="/home/inf148179/robust-cf/kdd/$EXPERIMENT_NAME/models/"
LOG_PATH="/home/inf148179/robust-cf/kdd/$EXPERIMENT_NAME/logs/"
RESULT_PATH="/home/inf148179/robust-cf/kdd/$EXPERIMENT_NAME/results/"

# SWEEP
# robust_method=[robx],[betarob]
datasets=[car_eval],[rice]
ex_type=[Architecture],[Bootstrap],[Seed]
base_cf=[gs],[dice],[face]
model_type_to_use=[neural_network]

echo "Running experiment: $EXPERIMENT_NAME"

source /home/inf148179/anaconda3/bin/activate
conda activate betarce

echo "Starting now ..."

python experiment_runner.py -cn $CONFIG_FILENAME --multirun \
    experiments_setup.posthoc_explainers=$robust_method \
    experiments_setup.classifiers=$model_type_to_use \
    experiments_setup.ex_types=$ex_type \
    experiments_setup.base_explainers=$base_cf \
    experiments_setup.datasets=$datasets \
    general.result_path=$RESULT_PATH \
    general.log_path=$LOG_PATH \
    general.model_path=$MODEL_PATH 
