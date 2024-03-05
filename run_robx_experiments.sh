CONFIGS_PATH=configs/experiment_correlation
ALPHAS=(060 065 070 075 080 085 090 093 095 097 099 09999)
DATASETS=("fico" "wine_quality" "breast_cancer")
EXPERIMENT_TYPES=("TwoSamplesOneDatasetExperimentData" "SameSampleExperimentData")

for DATASET in "${DATASETS[@]}"
do
    for EXPERIMENT_TYPE in "${EXPERIMENT_TYPES[@]}"
    do
        for ALPHA in "${ALPHAS[@]}"
        do
            CONFIG_BASENAME="config_a${ALPHA}.yml"
            CONFIG="$CONFIGS_PATH/$CONFIG_BASENAME"

            STOP_AFTER=1000
            ROBUST_METHOD=robx #choices: [statrob, robx, statrobxplus]
            MODEL_TYPE=mlp-torch #choices: [mlp-sklearn, rf-sklearn, mlp-torch]

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

            conda deactivate
        done
    done
done

echo "All experiments finished"
