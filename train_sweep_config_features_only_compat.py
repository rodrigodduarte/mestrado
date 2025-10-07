import os
import sys
import glob
import random
import subprocess
import shutil
import yaml
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
import wandb

from CustomFeaturesOnlyModel import CustomFeaturesOnlyModel
from features_datamodule import CustomFeaturesCSVModule, CustomFeaturesCSVModule_kf
from callbacks import EarlyStoppingAtSpecificEpoch, SaveBestOrLastModelCallback, EarlyStopCallback

# ================================
# Utilidades
# ================================

def empty_trash():
    trash_path = os.path.expanduser("~/.local/share/Trash")
    if os.path.exists(trash_path):
        subprocess.run(["rm", "-rf", f"{trash_path}/files/*", f"{trash_path}/info/*"], check=False)
        print("Lixeira esvaziada com sucesso no Linux.")


def load_hyperparameters(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def set_random_seeds(seed: int = 42):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ================================
# Resolução COMPATÍVEL de caminhos de CSV a partir do YAML original
# (não altera o .yaml do usuário)
# ================================

def _is_csv(path: str) -> bool:
    return isinstance(path, str) and path.lower().endswith('.csv') and os.path.isfile(path)


def _try_candidates(*cands):
    for p in cands:
        if p and os.path.isfile(p):
            return p
    return None


def resolve_csv_paths(hp: dict) -> tuple:
    """
    Resolve TRAIN_CSV e TEST_CSV sem exigir mudanças no YAML original.
    Regras:
      1) Se TRAIN_CSV/TEST_CSV já existem no YAML, usa-os direto.
      2) Caso contrário, usa TRAIN_DIR/TEST_DIR (do YAML de imagem) e tenta:
         - Se TRAIN_DIR/TEST_DIR apontarem para um arquivo .csv, usa direto.
         - Procurar dentro de cada diretório por arquivos padrão:
             features.csv, features_paths.csv, train.csv/test.csv
         - Procurar no diretório pai: train_features.csv/test_features.csv
      3) Se nada encontrado, lança erro explicando onde procurar arquivos.
    """
    # 1) Chaves diretas (se o usuário eventualmente adicionou)
    if 'TRAIN_CSV' in hp and 'TEST_CSV' in hp and _is_csv(hp['TRAIN_CSV']) and _is_csv(hp['TEST_CSV']):
        return hp['TRAIN_CSV'], hp['TEST_CSV']

    train_dir = hp.get('TRAIN_DIR')
    test_dir = hp.get('TEST_DIR')

    # 2) Se apontarem para arquivos .csv, aceitar
    if _is_csv(train_dir) and _is_csv(test_dir):
        return train_dir, test_dir

    # 3) Se forem diretórios, testar candidatos comuns
    def cand_for(d):
        if not isinstance(d, str):
            return None
        if os.path.isfile(d) and d.lower().endswith('.csv'):
            return d
        if not os.path.isdir(d):
            return None
        parent = os.path.dirname(d.rstrip('/'))
        cands = [
            os.path.join(d, 'features.csv'),
            os.path.join(d, 'features_paths.csv'),
            os.path.join(d, 'train.csv'),
            os.path.join(d, 'test.csv'),  # útil se o usuário apontou TEST_DIR
            os.path.join(parent, 'train_features.csv') if 'train' in os.path.basename(d) else os.path.join(parent, 'test_features.csv'),
        ]
        return _try_candidates(*cands)

    train_csv = cand_for(train_dir)
    test_csv = cand_for(test_dir)

    if train_csv and test_csv:
        return train_csv, test_csv

    # 4) fallback: buscar por *.csv dentro do diretório
    def any_csv_in(d):
        if isinstance(d, str) and os.path.isdir(d):
            hits = sorted(glob.glob(os.path.join(d, '*.csv')))
            return hits[0] if hits else None
        return None

    train_csv = train_csv or any_csv_in(train_dir)
    test_csv = test_csv or any_csv_in(test_dir)

    if train_csv and test_csv:
        return train_csv, test_csv

    raise FileNotFoundError(
        "Não foi possível localizar os CSVs de features com base no YAML fornecido.\n"
        f"TREINOS: tente colocar um 'features.csv' dentro de {train_dir}\n"
        f"TESTE:   tente colocar um 'features.csv' dentro de {test_dir}\n"
        "Ou acrescente explicitamente TRAIN_CSV e TEST_CSV no YAML apontando para os arquivos."
    )


# ================================
# Treino (PIPELINE DO SWEEP) — FEATURES ONLY, COMPATÍVEL COM YAML ORIGINAL
# ================================

def train_model(config=None):
    hp = load_hyperparameters('config.yaml')

    # Resolve CSVs a partir do YAML original (sem alterá-lo)
    train_csv, test_csv = resolve_csv_paths(hp)

    # Escolha entre K-Fold (chave original K_FOLDS) e split simples
    use_kf = bool(hp.get('K_FOLDS', 0)) and int(hp['K_FOLDS']) > 1

    with wandb.init(project=hp['PROJECT'], config=config):
        cfg = wandb.config

        # DataModule
        if use_kf:
            data_module = CustomFeaturesCSVModule_kf(
                train_csv=train_csv,
                test_csv=test_csv,
                batch_size=hp['BATCH_SIZE'],
                num_workers=hp['NUM_WORKERS'],
                n_splits=int(hp['K_FOLDS']),
                fold_idx=int(hp.get('FOLD_IDX', 0)),
                seed=int(hp.get('SEED', 42))
            )
        else:
            data_module = CustomFeaturesCSVModule(
                train_csv=train_csv,
                test_csv=test_csv,
                batch_size=hp['BATCH_SIZE'],
                num_workers=hp['NUM_WORKERS'],
                val_split=float(hp.get('VAL_SPLIT', 0.2)),
                seed=int(hp.get('SEED', 42))
            )

        # Modelo (features-only) — mantém nomes do YAML original
        model = CustomFeaturesOnlyModel(
            tmodel=hp.get('TMODEL', 'vector_only'),
            name_dataset=hp.get('NAME_DATASET', 'features'),
            shape=tuple(hp.get('SHAPE', (0, 0))),
            epochs=int(hp['MAX_EPOCHS']),
            learning_rate=float(cfg.get('learning_rate', hp.get('LEARNING_RATE', 1e-3))),
            features_dim=int(hp['FEATURES_DIM']),
            drop_path_rate=float(cfg.get('drop_path_rate', hp.get('DROP_PATH_RATE', 0.0))),  # aceito e ignorado
            num_classes=int(hp['NUM_CLASSES']),
            label_smoothing=float(cfg.get('label_smoothing', hp.get('LABEL_SMOOTHING', 0.0))),
            optimizer_momentum=(float(cfg.get('optimizer_momentum', hp.get('OPTIMIZER_MOMENTUM', 0.9))), 0.999),
            weight_decay=float(cfg.get('weight_decay', hp.get('WEIGHT_DECAY', 0.0))),
            layer_scale=float(cfg.get('layer_scale', hp.get('LAYER_SCALE', 1.0)))
        )

        # Logger e callbacks
        wandb_logger = WandbLogger(project=hp['PROJECT'])
        run_name = wandb.run.name
        ckpt_dir = hp['CHECKPOINT_PATH']
        os.makedirs(ckpt_dir, exist_ok=True)
        checkpoint_path = os.path.join(ckpt_dir, f"{run_name}.ckpt")

        save_model_cb = SaveBestOrLastModelCallback(checkpoint_path)
        epoch_cb = EarlyStoppingAtSpecificEpoch(
            patience=2,
            threshold=1e-3,
            monitor="val_loss",
            mode="min",
            verbose=True
        )
        early_stop_cb = EarlyStopCallback(
            metric_name="val_loss",
            threshold=0.5,
            target_epoch=3
        )

        trainer = pl.Trainer(
            logger=wandb_logger,
            log_every_n_steps=10,
            accelerator=hp['ACCELERATOR'],
            devices=hp['DEVICES'],
            precision=hp['PRECISION'],
            max_epochs=int(hp['MAX_EPOCHS']),
            callbacks=[TQDMProgressBar(leave=True), save_model_cb, epoch_cb, early_stop_cb]
        )

        # Treino + Teste
        trainer.fit(model, data_module)

        # Carrega melhor checkpoint salvo
        model = CustomFeaturesOnlyModel.load_from_checkpoint(checkpoint_path)
        trainer.test(model, data_module)

        # Limpeza (mantém filosofia do molde, sem tocar no YAML)
        if os.path.exists(ckpt_dir):
            for file_name in os.listdir(ckpt_dir):
                file_path = os.path.join(ckpt_dir, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"Todos os arquivos foram removidos da pasta {ckpt_dir}.")
        empty_trash()

        # Atenção: no molde original havia remoção de uma pasta com nome PROJECT.
        # Para evitar apagar diretórios do usuário, não removemos nada aqui além dos checkpoints.

        wandb.finish()


if __name__ == '__main__':
    # Login e seeds
    wandb.login()
    set_random_seeds(int(os.environ.get('SEED', 42)))

    # Sweep config (mantém estilo do molde, mas focado para MLP de features)
    sweep_config = {
        'method': 'random',
        'metric': {'name': 'val_loss', 'goal': 'minimize'},
        'parameters': {
            'learning_rate': {'min': 1e-5, 'max': 3e-3, 'distribution': 'uniform'},
            'weight_decay':   {'min': 0.0, 'max': 2e-1, 'distribution': 'uniform'},
            'optimizer_momentum': {'min': 0.85, 'max': 0.99, 'distribution': 'uniform'},
            'layer_scale':    {'min': 0.5, 'max': 4.5, 'distribution': 'uniform'},
            'drop_path_rate': {'min': 0.0, 'max': 0.5, 'distribution': 'uniform'},  # ignorado no modelo
            'label_smoothing':{'min': 0.0, 'max': 0.2, 'distribution': 'uniform'}
        }
    }

    # Carrega YAML apenas para obter PROJECT e SWEEP_COUNT (se houver)
    hp = load_hyperparameters('config.yaml')
    sweep_id = wandb.sweep(sweep_config, project=hp['PROJECT'])
    wandb.agent(sweep_id, function=train_model, count=hp.get('SWEEP_COUNT', 200))
    wandb.finish()
