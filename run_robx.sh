CONFIGS=(config06.yml config07.yml config08.yml config09.yml)

for CONFIG in ${CONFIGS[@]}
do


    DATASET=fico #choices: [fico, german]
    STOP_AFTER=200
    ROBUST_METHOD=robx #choices: [statrob, robx, statrobxplus]
    EXPERIMENT_TYPE=SameSampleExperimentData #choices: [TwoSamplesOneDatasetExperimentData, SameSampleExperimentData]
    MODEL_TYPE=mlp-torch #choices: [mlp-sklearn, rf-sklearn, mlp-torch]

    # Extract the first part of the config name without the extension
    CONFIG_BASENAME="${CONFIG%%.*}"

    # Construct the NAME variable
    NAME="torch-$DATASET-$ROBUST_METHOD-var01-$CONFIG_BASENAME"

    echo "Running experiment with name: $NAME"


    source activate robustcf


    python src/run_experiments.py --dataset $DATASET \
                                    --experiment $NAME  \
                                    --stop_after $STOP_AFTER \
                                    --config $CONFIG \
                                    --robust_method $ROBUST_METHOD \
                                    --experiment_type $EXPERIMENT_TYPE \
                                    --model_type $MODEL_TYPE

    source deactivate

done
echo "All experiments finished"

