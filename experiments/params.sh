EXPERIMENT_NAME="params"
CONFIG_FILENAME="config_ex_params"

# PATHS
BASE_PATH="/home/inf148179/robust-cf/kdd/$EXPERIMENT_NAME/"
MODEL_PATH="/home/inf148179/robust-cf/kdd/$EXPERIMENT_NAME/models/"
LOG_PATH="/home/inf148179/robust-cf/kdd/$EXPERIMENT_NAME/logs/"
RESULT_PATH="/home/inf148179/robust-cf/kdd/$EXPERIMENT_NAME/results/"

# SWEEP
e2e_explainers=[]

echo "Running experiment: $EXPERIMENT_NAME"

source /home/inf148179/anaconda3/bin/activate
conda activate betarce

echo "Starting now ..."

python experiment_runner.py -cn $CONFIG_FILENAME --multirun \
    experiments_setup.e2e_explainers=$e2e_explainers \
    general.result_path=$RESULT_PATH \
    general.log_path=$LOG_PATH \
    general.model_path=$MODEL_PATH 
