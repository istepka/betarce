

DATASET=wine_quality #choices: [fico, german, wine_quality, breast_cancer]
STOP_AFTER=50
CONFIG=config09.yml
ROBUST_METHOD=statrob #choices: [statrob, robx, statrobxplus]
EXPERIMENT_TYPE=SameSampleExperimentData #choices: [TwoSamplesOneDatasetExperimentData, SameSampleExperimentData]
MODEL_TYPE=mlp-torch #choices: [mlp-sklearn, rf-sklearn, mlp-torch]

# Extract the first part of the config name without the extension
CONFIG_BASENAME="${CONFIG%%.*}"

# Construct the NAME variable
NAME="$MODEL_TYPE-$DATASET-$ROBUST_METHOD-$EXPERIMENT_TYPE-$CONFIG_BASENAME"

echo "Running experiment with name: $NAME"


source activate robustcf


python src/run_experiments.py --dataset $DATASET \
                                --experiment $NAME  \
                                --stop_after $STOP_AFTER \
                                --config $CONFIG \
                                --robust_method $ROBUST_METHOD \
                                --experiment_type $EXPERIMENT_TYPE \
                                --model_type $MODEL_TYPE


echo "All experiments finished"

