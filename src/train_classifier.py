import pandas as pd
import numpy as np
import catboost as cb
import pickle
import hydra
from omegaconf import DictConfig
import logging
import os
import optuna

from utils.fe_for_catboost import prepare_classification_data, split_by_block_id, resample_training_data

log = logging.getLogger(__name__)


def objective(trial: optuna.Trial, X_train, y_train, X_val, y_val, cfg):
    params = {
        'iterations': cfg.model.params.iterations,
        'loss_function': 'MultiClass',
        'eval_metric': 'TotalF1:average=Weighted',
        'random_seed': 42,
        'verbose': 100,
        'depth': trial.suggest_int('depth', 6, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        #'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
        'task_type': 'GPU',
    }

    model = cb.CatBoostClassifier(**params, auto_class_weights='Balanced')
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=cfg.hparams.early_stopping_patience
    )
    
    return model.get_best_score()['validation']['TotalF1:average=Weighted']

@hydra.main(config_path="../conf", config_name="train_classifier", version_base=None)
def train_classifier(cfg: DictConfig) -> None:
    log.info("="*50)
    log.info("Запуск пайплайна обучения КЛАССИФИКАТОРА...")
    log.info("="*50)

    log.info("--- Шаг 1/4: Загрузка и подготовка данных ---")
    csv_path = os.path.join(cfg.data.root_path, cfg.data.data_path)
    df_raw = pd.read_csv(csv_path, header=None)
 
    df_raw.columns = [
        "block_id", "frame_idx", "E_mu_Z", "E_mu_phys_est", "E_mu_X", "E_nu1_X", "E_nu2_X", "E_nu1_Z", "E_nu2_Z",
        "N_mu_X", "M_mu_XX", "M_mu_XZ", "M_mu_X", "N_mu_Z", "M_mu_ZZ", "M_mu_Z", "N_nu1_X", "M_nu1_XX",
        "M_nu1_XZ", "M_nu1_X", "N_nu1_Z", "M_nu1_ZZ", "M_nu1_Z", "N_nu2_X", "M_nu2_XX", "M_nu2_XZ",
        "M_nu2_X", "N_nu2_Z", "M_nu2_ZZ", "M_nu2_Z", "nTot", "bayesImVoltage", "opticalPower",
        "polarizerVoltages[0]", "polarizerVoltages[1]", "polarizerVoltages[2]", "polarizerVoltages[3]",
        "temp_1", "biasVoltage_1", "temp_2", "biasVoltage_2", "synErr", "N_EC_rounds",
        "maintenance_flag", "estimator_name", "f_EC", "E_mu_Z_est", "R", "s", "p",
    ]
    df_raw = df_raw.rename(columns={"block_id": "id", "frame_idx": "date"})

    X, y, artifacts = prepare_classification_data(df_raw, target_col=cfg.model.target_column)

    original_indices = X.index
    X['id'] = df_raw.loc[original_indices, 'id'].values

    X_train, X_val, y_train, y_val = split_by_block_id(X, y, test_size=cfg.data.val_size)
    log.info(f"Данные разделены. Train: {X_train.shape}, Validation: {X_val.shape}")

    X_train_resampled, y_train_resampled = resample_training_data(X_train, y_train)

    log.info(f"Распределение классов в y_train:\n{y_train.value_counts(normalize=True)}")
    log.info(f"Распределение классов в y_val:\n{y_val.value_counts(normalize=True)}")
    
    # log.info("--- Шаг 2/4: Инициализация и обучение CatBoostClassifier ---")
    
    # class_counts = y_train.value_counts()
    # class_weights = len(y_train) / (len(class_counts) * class_counts)

    # model = cb.CatBoostClassifier(
    #     **cfg.model.params,
    #     #class_weights=class_weights.to_dict(),
    #     verbose=100
    # )
    
    # model.fit(
    #     X_train_resampled, y_train_resampled,
    #     eval_set=(X_val, y_val),
    #     early_stopping_rounds=cfg.hparams.early_stopping_patience,
    # )
    log.info("--- Шаг 2a/4: Запуск подбора гиперпараметров с Optuna ---")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train_resampled, y_train_resampled, X_val, y_val, cfg), 
                   n_trials=cfg.hparams.optuna_trials)

    best_params = study.best_params
    log.info(f"Лучшие параметры найдены: {best_params}")

    log.info("--- Шаг 2b/4: Обучение финальной модели на лучших параметрах ---")
    final_params = cfg.model.params
    final_params.update(best_params) # Обновляем конфиг лучшими параметрами

    model = cb.CatBoostClassifier(**final_params, auto_class_weights='Balanced', verbose=100)
    model.fit(
        X_train_resampled, y_train_resampled,
        eval_set=(X_val, y_val),
        early_stopping_rounds=cfg.hparams.early_stopping_patience
    )


    log.info("--- Шаг 3/4: Сохранение артефактов ---")
    model.save_model("catboost_model.cbm")
    
    r_categories = pd.Categorical(df_raw[cfg.model.target_column].dropna()).categories
    index_to_r_map = dict(enumerate(r_categories))
    
    artifacts['index_to_r_map'] = index_to_r_map
    
    r_to_s_map = df_raw.groupby('R')['s'].median().to_dict()
    r_to_e_mu_z_map = df_raw.groupby('R')['E_mu_Z'].median().to_dict()
    artifacts['r_to_s_map'] = r_to_s_map
    artifacts['r_to_e_mu_z_map'] = r_to_e_mu_z_map
    
    with open("classifier_artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)
    
    log.info(f"Артефакты сохранены в: {os.getcwd()}")

    log.info("--- Шаг 4/4: Важность признаков ---")
    feature_importance = pd.DataFrame({
        'feature': model.feature_names_,
        'importance': model.get_feature_importance()
    }).sort_values('importance', ascending=False)
    
    log.info(f"Топ-15 важных признаков:\n{feature_importance.head(15)}")

    log.info("="*50)
    log.info("Пайплайн обучения классификатора успешно завершен!")
    log.info("="*50)


if __name__ == '__main__':
    train_classifier()