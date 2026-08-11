import os


class RankConfig:
    # Negatives
    N_NEGATIVES           = 50
    NEGATIVE_SEED         = 42
    # ECFP4
    ECFP4_RADIUS          = 2
    ECFP4_NBITS           = 2048
    # LambdaMART
    N_ESTIMATORS          = 100
    LEARNING_RATE         = 0.1
    NUM_LEAVES            = 31
    MIN_CHILD_SAMPLES     = 20
    EARLY_STOPPING_ROUNDS = 0
    NDCG_EVAL_AT          = [1, 5, 10]
    # Output
    RANK_MODELS_DIR       = os.path.join("output", "rank_models")
