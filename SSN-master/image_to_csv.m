% Diretório de entrada
inputDir = '/home/rodrigoduarte/Documentos/projeto/imagens/soja';
% Diretório de saída
outputDir = '/home/rodrigoduarte/Documentos/projeto/imagens/soja';

fprintf('[INFO] inputDir  = %s\n', inputDir);
fprintf('[INFO] outputDir = %s\n', outputDir);
if ~exist(inputDir, 'dir')
    fprintf('[WARN] inputDir NÃO existe: %s\n', inputDir);
end
if ~exist(outputDir, 'dir')
    fprintf('[WARN] outputDir NÃO existe: %s\n', outputDir);
end

% % Criar o diretório de saída se não existir
% if ~exist(outputDir, 'dir')
%     mkdir(outputDir);
% end

% Listar os diretórios de treino e teste
subDirs = {'train', 'test'};
fprintf('[INFO] subDirs = {%s, %s}\n', subDirs{1}, subDirs{2});

for i = 1:length(subDirs)
    currSubDir = fullfile(inputDir, subDirs{i});
    fprintf('\n[INFO] Processando subdir: %s\n', currSubDir);
    if ~exist(currSubDir, 'dir')
        fprintf('[WARN] Subdir NÃO existe: %s\n', currSubDir);
    end
    
    % Listar todas as classes no diretório atual (treino ou teste)
    classDirs = dir(currSubDir);
    classDirs = classDirs([classDirs.isdir] & ~ismember({classDirs.name}, {'.', '..'}));
    fprintf('[INFO] %d classes encontradas em %s\n', numel(classDirs), currSubDir);
    
    for j = 1:length(classDirs)
        classDir = fullfile(currSubDir, classDirs(j).name);
        fprintf('[INFO] Classe: %s\n', classDir);

        % Criar o diretório correspondente no diretório de saída
        outputClassDir = fullfile(outputDir, subDirs{i}, classDirs(j).name);
        fprintf('[INFO] Pasta de saída da classe: %s\n', outputClassDir);
        
        if ~exist(outputClassDir, 'dir')
            fprintf('[INFO] Criando diretório: %s\n', outputClassDir);
            mkdir(outputClassDir);
        end
        
        % Listar todas as imagens na classe
        pattern = fullfile(classDir, '*.jpg');
        fprintf('[INFO] Procurando imagens com padrão: %s\n', pattern);
        imageFiles = dir(pattern);
        fprintf('[INFO] %d imagens encontradas em %s\n', numel(imageFiles), classDir);
        if isempty(imageFiles)
            fprintf('[WARN] NENHUMA imagem *.jpg em: %s\n', classDir);
        end
        
        for k = 1:length(imageFiles)
            % Lê a imagem
            imagePath = fullfile(classDir, imageFiles(k).name);
            fprintf('[INFO] Lendo imagem (%d/%d): %s\n', k, numel(imageFiles), imagePath);
            image = imread(imagePath);
            fprintf('[INFO]   Tam. original: %dx%dx%d | classe: %s\n', size(image,1), size(image,2), size(image,3), class(image));
            
            % % Redimensiona a imag em
            image = imresize(image, [224, 224]);
            fprintf('[INFO]   Redimensionada para: %dx%dx%d\n', size(image,1), size(image,2), size(image,3));
            
            % Aplica a função SSN
            fprintf('[INFO]   Executando SSN(image, 6)\n');
            features = SSN(image, 6); % Altere o segundo parâmetro conforme necessário
            fprintf('[INFO]   SSN OK | size(features) = [%d %d]\n', size(features,1), size(features,2));
            
            % Salva as features em um arquivo CSV com o sufixo '_csv'
            [~, name, ~] = fileparts(imageFiles(k).name);
            outputFilePath = fullfile(outputClassDir, [name, '.csv']);
            fprintf('[INFO]   Salvando CSV: %s\n', outputFilePath);
            
            % Escreve as características em um arquivo CSV
            writematrix(features, outputFilePath);
            fprintf('[OK]     CSV salvo.\n');
        end
    end
end

fprintf('\n[DONE] Finalizado.\n');
