CONFIGS_PATH=configs/experiment_correlation
ALPHAS=(060 065 070 075 080 090 095 099 09999)
DATASETS=("fico" "wine_quality" "breast_cancer")
EXPERIMENT_TYPES=("TwoSamplesOneDatasetExperimentData" "SameSampleExperimentData")
BASE_CF_METHODS=("gs" "dice")

for BASE_CF_METHOD in "${BASE_CF_METHODS[@]}"
do
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
                NAME="$MODEL_TYPE-$DATASET-$ROBUST_METHOD-$EXPERIMENT_TYPE-$BASE_CF_METHOD-$CONFIG_BASENAME"

                echo "Running experiment with name: $NAME"

                source activate robustcf

                python src/run_experiments.py --dataset $DATASET \
                                                --experiment $NAME  \
                                                --stop_after $STOP_AFTER \
                                                --config $CONFIG \
                                                --robust_method $ROBUST_METHOD \
                                                --experiment_type $EXPERIMENT_TYPE \
                                                --model_type $MODEL_TYPE \
                                                --base_cf_method $BASE_CF_METHOD

                conda deactivate
            done
        done
    done
done

echo "All experiments finished"
