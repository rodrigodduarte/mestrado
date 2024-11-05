clear;
mex -setup C++;
mex SSN_getFeatureMaps.cpp
mex SSNgrey_getFeatureMaps.cpp

extension = '*.png';  % Define a extensão como .png
dataset = ["train", "test"];
base_path = "/home/rodrigo/Documentos/mestrado/GitHub/imagens/swedish/imagens/";
output_base_path = "/home/rodrigo/Documentos/mestrado/GitHub/imagens_csv/swedish/";

% Cria os diretórios de saída de acordo com a estrutura de diretórios do dataset
for di = dataset
    data_dir = fullfile(base_path, di);
    classes = dir(data_dir);
    classes = classes([classes.isdir] & ~ismember({classes.name}, {'.', '..'}));
    for class = classes'
        mkdir(fullfile(output_base_path, string(di), string(class.name)));
    end
end

% Processamento das imagens e geração dos arquivos CSV
for ni = 2:12
    for di = dataset
        data_dir = fullfile(base_path, di);
        classes = dir(data_dir);
        classes = classes([classes.isdir] & ~ismember({classes.name}, {'.', '..'}));

        % Loop pelas classes (subdiretórios)
        for class = classes'
            path = fullfile(data_dir, class.name);
            files = dir(fullfile(path, extension));

            for file = files'
                % Extrai características e salva no array A
                A = [];
                A = [A SSN(fullfile(file.folder, file.name), ni)];
                A = [A string(file.name)];
                A = [A upper(class.name)];

                % Gera o nome do arquivo CSV com o mesmo nome da imagem e "_csv" ao final
                [~, name, ~] = fileparts(file.name);
                output_name = fullfile(output_base_path, string(di), string(class.name), strcat(name, "_csv.csv"));
                
                % Salva as características extraídas no arquivo CSV
                writematrix(A, output_name);
            end
        end
    end
    strcat("OK: ", string(ni))
end
