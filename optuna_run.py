"""
This file handles the optuna hyperparameter tuning by combining BiHDNCE.py and Evaluation.py
"""

import logging
import optuna
import os
import shutil
from BiHDNCE import optuna_train
from Evaluation import optuna_evaluate


num_epoch = 8

def objective(trial):
    # TODO: check the server gpu memory and adjust the batch size accordingly
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256])
    lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
    max_seq_length = trial.suggest_categorical("max_seq_length", [8, 16, 32, 64, 128, 256])

    logging.info("Deleting and copying models: ")
    # Step 1: Check if the "models/bihardnce" directory exists
    if os.path.exists("models/bihardnce"):
        # Step 2: If the directory exists, delete it
        shutil.rmtree("models/bihardnce")

    # Step 3: Create a new directory named "bihardnce" inside the "models" directory (models/bihardnce)
    os.mkdir("models/bihardnce")

    # Step 4: Copy all the files under the folder "models/bert" to the folder "models/bihardnce"
    shutil.copytree("models/bert", "models/bihardnce")

    logging.info("Starting optuna hyperparameter tuning: ")
    for i in range(num_epoch):
        logging.info("Epoch: " + str(i))
        # optuna_train
        optuna_train('bihardnce', i, batch_size, lr, max_seq_length)
        # optuna_evaluate
        micro_f1_num = optuna_evaluate('bihardnce', i)

    return micro_f1_num

# hyperparameters to tune: learning rate, max_seq_length, train_batch_size
# other hyperparameters: alpha, beta, sampling_method, num_hard_negative_queries, num_hard_negative_candidates, positive_threshold
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
