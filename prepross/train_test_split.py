import os
import shutil
from sklearn.model_selection import train_test_split

# Caminho para o diretório contendo suas classes com imagens
caminho_diretorio_original = 'imagens/flavia_t'
caminho_diretorio_novo = 'imagens/flavia'

# Lista de todas as classes no diretório original
classes = os.listdir(caminho_diretorio_original)

# Criar diretórios de treino e teste no novo diretório
os.makedirs(os.path.join(caminho_diretorio_novo, 'train'), exist_ok=True)
os.makedirs(os.path.join(caminho_diretorio_novo, 'test'), exist_ok=True)

# Para cada classe, dividir as imagens em conjuntos de treino e teste
for classe in classes:
    caminho_classe_original = os.path.join(caminho_diretorio_original, classe)
    
    # Lista de todas as imagens na classe
    imagens = os.listdir(caminho_classe_original)
    
    # Dividir as imagens em conjuntos de treino e teste (80% treino, 20% teste)
    treino, teste = train_test_split(imagens, test_size=0.2, random_state=42)
    
    # Criar diretórios de treino e teste para a classe no novo diretório
    diretorio_treino_novo = os.path.join(caminho_diretorio_novo, 'train', classe)
    diretorio_teste_novo = os.path.join(caminho_diretorio_novo, 'test', classe)

    os.makedirs(diretorio_treino_novo, exist_ok=True)
    os.makedirs(diretorio_teste_novo, exist_ok=True)
    
    # Mover as imagens para os diretórios correspondentes no novo diretório
    for imagem in treino:
        shutil.copy(os.path.join(caminho_classe_original, imagem), os.path.join(diretorio_treino_novo, imagem))
    
    for imagem in teste:
        shutil.copy(os.path.join(caminho_classe_original, imagem), os.path.join(diretorio_teste_novo, imagem))
