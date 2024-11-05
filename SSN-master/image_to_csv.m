% Diretório de entrada
inputDir = '/home/rodrigo/Documentos/mestrado/GitHub/imagens/swedish/';
% Diretório de saída
outputDir = '/home/rodrigo/Documentos/mestrado/GitHub/imagens_csv/swedish/';

% Criar o diretório de saída se não existir
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Listar os diretórios de treino e teste
subDirs = {'train', 'test'};

for i = 1:length(subDirs)
    currSubDir = fullfile(inputDir, subDirs{i});
    
    % Listar todas as classes no diretório atual (treino ou teste)
    classDirs = dir(currSubDir);
    classDirs = classDirs([classDirs.isdir] & ~ismember({classDirs.name}, {'.', '..'}));
    
    for j = 1:length(classDirs)
        classDir = fullfile(currSubDir, classDirs(j).name);
        % Criar o diretório correspondente no diretório de saída
        outputClassDir = fullfile(outputDir, subDirs{i}, classDirs(j).name);
        
        if ~exist(outputClassDir, 'dir')
            mkdir(outputClassDir);
        end
        
        % Listar todas as imagens na classe
        imageFiles = dir(fullfile(classDir, '*.png'));
        
        for k = 1:length(imageFiles)
            % Lê a imagem
            imagePath = fullfile(classDir, imageFiles(k).name);
            image = imread(imagePath);
            
            % Redimensiona a imagem
            image = imresize(image, [224, 224]);
            
            % Aplica a função SSN
            features = SSN(image, 12); % Altere o segundo parâmetro conforme necessário
            
            % Salva as features em um arquivo CSV com o sufixo '_csv'
            [~, name, ~] = fileparts(imageFiles(k).name);
            outputFilePath = fullfile(outputClassDir, [name, '_csv.csv']);
            
            % Escreve as características em um arquivo CSV
            writematrix(features, outputFilePath);
        end
    end
end
