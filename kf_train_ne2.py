import os
import shutil
import torch
import pytorch_lightning as pl
import numpy as np
import wandb
import yaml
import random
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from model import CustomModel
from kf_data import CustomImageModule_kf
from callbacks import (
    EarlyStoppingAtSpecificEpoch,
    SaveBestOrLastModelCallback,
    EarlyStopCallback

)

# Carregar hiperparâmetros do arquivo config2.yaml
def load_hyperparameters(file_path):
    with open(file_path, 'r') as file:
        hyperparams = yaml.safe_load(file)
    return hyperparams

# Configurar sementes para garantir reprodutibilidade
def set_random_seeds():
    torch.backends.cudnn.deterministic = True
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Função principal para treinamento com validação cruzada
def train_model(config=None):
    hyperparams = load_hyperparameters('config2.yaml')
    k_splits = hyperparams['K_FOLDS']
    best_checkpoint_path = None
    epochs_per_fold = hyperparams['MAX_EPOCHS'] // k_splits  
    
    
    with wandb.init(project=hyperparams["PROJECT"], config=config):
        print(wandb.run.name)
        config_sweep = wandb.config
        
        # Configurar o modelo
        model = CustomModel(
            tmodel=hyperparams["TMODEL"],
            name_dataset= hyperparams["NAME_DATASET"],
            epochs=hyperparams['MAX_EPOCHS'],
            shape=hyperparams["SHAPE"],                              # Fixo
            learning_rate=float(config_sweep.learning_rate),       # Variável do sweep
            scale_factor=hyperparams['SCALE_FACTOR'],       # Fixo
            drop_path_rate=config_sweep.drop_path_rate,   # Fixo
            num_classes=hyperparams['NUM_CLASSES'],         # Fixo
            label_smoothing=config_sweep.label_smoothing,
            optimizer_momentum=(config_sweep.optimizer_momentum, 0.999)  # Fixo
        )
        
        stop_all_folds_callback = EarlyStopCallback(metric_name="val_loss", threshold=0.7, target_epoch=4)
        
        wandb_logger = WandbLogger(project=hyperparams["PROJECT"])

        run_name = wandb.run.name
        checkpoint_path = f"{hyperparams['CHECKPOINT_PATH']}/{run_name}.ckpt"

        for fold in range(k_splits):
    
            # if stop_all_folds_callback.should_stop_training():
            #     print("🚨 Stop All Folds foi ativado! Encerrando a execução e iniciando nova run.")
            #     break  # Sai do treinamento antes de começar os próximos folds         
        
            print(f"\nTreinando Fold {fold+1}/{k_splits}")

            # Configurar o DataModule
            data_module = CustomImageModule_kf(
                train_dir=hyperparams['TRAIN_DIR'],
                test_dir=hyperparams['TEST_DIR'],
                shape=hyperparams['SHAPE'],
                batch_size=hyperparams['BATCH_SIZE'],
                num_workers=hyperparams['NUM_WORKERS']
            )
            data_module.setup(stage='fit')

            callbacks = [
                TQDMProgressBar(leave=True),
                SaveBestOrLastModelCallback(checkpoint_path),
                # EarlyStoppingAtSpecificEpoch(patience=4, threshold=1e-3, monitor="val_loss"),
                stop_all_folds_callback
            ]



            trainer = pl.Trainer(
                logger=wandb_logger,
                log_every_n_steps=10,
                accelerator=hyperparams['ACCELERATOR'],
                devices=hyperparams['DEVICES'],
                precision=hyperparams['PRECISION'],
                max_epochs=epochs_per_fold,
                callbacks=callbacks
            )

            trainer.fit(model, data_module)
            
            best_checkpoint_path = checkpoint_path

            if stop_all_folds_callback.should_stop_training():
                print("🚨 Stop All Folds foi ativado! Encerrando a execução e iniciando nova run.")
                break  # Sai do treinamento antes de começar os próximos folds     

            else:
                print(f"\nTreinamento finalizado. Modelo salvo em: {best_checkpoint_path}")

        test_accuracy = 0
        
        if best_checkpoint_path and not stop_all_folds_callback.should_stop_training():
            print("\nSalvando o melhor modelo antes de carregar para o teste...")

            # 🔹 Definir diretório de destino e salvar o modelo diretamente lá
            final_model_dir = f"{hyperparams['NAME_DATASET']}_bestmodel/{wandb.run.name}"
            os.makedirs(final_model_dir, exist_ok=True)
            final_model_path = os.path.join(final_model_dir, "best_model.ckpt")

            # 🔹 Copiar o melhor modelo salvo pelo callback para a pasta final
            shutil.copy(best_checkpoint_path, final_model_path)

            print(f"Melhor modelo copiado para: {final_model_path}")

            # 🔹 Agora carregamos o modelo salvo para garantir que está correto
            print("\nIniciando teste final no melhor modelo...")
            best_model = CustomModel.load_from_checkpoint(final_model_path)

            data_module.setup(stage='test')
            test_results = trainer.test(best_model, data_module)

            test_accuracy = test_results[0].get("test_accuracy", 0)  # Obtém a métrica de teste

            print(f"Teste final concluído com sucesso usando {final_model_path}")



        # 🔹 Determinar o diretório onde estava salvo o checkpoint anterior
        best_checkpoint_dir = os.path.dirname(best_checkpoint_path)  # Obtém o diretório do melhor modelo salvo

        
        # Se a acurácia de teste for 100%, interrompe o Sweep
        if test_accuracy >= 1.0 and best_checkpoint_path and not stop_all_folds_callback.should_stop_training():
            print("🚨 Acurácia de 100% atingida! Interrompendo o Sweep do WandB.")
            wandb.finish()  # Finaliza a execução da `run`
            # wandb.api.stop_sweep(sweep_id) # 🔥 Para o Sweep programaticamente

        
        # 🔹 Remover o diretório do checkpoint anterior, se existir
    if os.path.exists(best_checkpoint_dir):
        print(f"Removendo todos os arquivos do diretório anterior: {best_checkpoint_dir}...")
        
        try:
            shutil.rmtree(best_checkpoint_dir)  # Remove a pasta inteira
            print(f"Diretório de checkpoints removido: {best_checkpoint_dir}")
        except Exception as e:
            print(f"Erro ao deletar {best_checkpoint_dir}: {e}")
    else:
        print(f"O diretório {best_checkpoint_dir} não existe, nada a remover.")
        
    wandb.finish()

if __name__ == "__main__":
    set_random_seeds()
    # sweep_config = {
    #     'method': 'random',
    #     'metric': {'name': 'val_loss', 'goal': 'minimize'},
    #     'parameters': {
    #         'learning_rate': {'min': 6e-6, 'max': 1e-4, 'distribution': 'uniform'},
    #         'weight_decay': {'min': 1e-7, 'max': 1e-6, 'distribution': 'uniform'},
    #         'optimizer_momentum': {'min': 0.92, 'max': 0.99, 'distribution': 'uniform'},
    #         'mlp_vector_model_scale': {'min': 0.8, 'max': 1.3, 'distribution': 'uniform'},
    #         'layer_scale': {'min': 0.75, 'max': 3, 'distribution': 'uniform'},
    #         'drop_path_rate': {'min': 0.0, 'max': 0.5, 'distribution': 'uniform'},
    #         'label_smoothing': {'min': 0.0, 'max': 0.2, 'distribution': 'uniform'}
    #     }
    # }
    sweep_config = {
        'method': 'random',
        'metric': {'name': 'val_loss', 'goal': 'minimize'},
        'parameters': {
            'learning_rate': {'min': 0.00016246, 'max':0.00016247, 'distribution': 'uniform'},
            'weight_decay': {'min': 4.4776e-7, 'max': 4.4777e-7, 'distribution': 'uniform'},
            'optimizer_momentum': {'min': 0.9257, 'max': 0.9258, 'distribution': 'uniform'},
            'mlp_vector_model_scale': {'min': 1.07551, 'max': 1.07552, 'distribution': 'uniform'},
            'layer_scale': {'min': 0.93523, 'max': 0.93524, 'distribution': 'uniform'},
            'drop_path_rate': {'min': 0.11709, 'max': 0.11710, 'distribution': 'uniform'},
            'label_smoothing': {'min': 0.011871, 'max': 0.011872, 'distribution': 'uniform'}
        }
    }
    sweep_id = wandb.sweep(sweep_config, project=load_hyperparameters('config2.yaml')["PROJECT"])
    wandb.agent(sweep_id, function=train_model, count=200)
    wandb.finish()
