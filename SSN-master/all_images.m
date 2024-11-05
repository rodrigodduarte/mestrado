clear;
mex -setup C++;
mex SSN_getFeatureMaps.cpp
mex SSNgrey_getFeatureMaps.cpp

extension = '*.jpg';

dataset = ["train" "eval"];
target = ["paracatuana" "florida"];
imgType = ["pro" "neg"];

for di=dataset
    mkdir(string(di))
end

for ni=2:12
    for di=dataset
        A = [];
        for ti=target
            for ii=imgType
                path = strcat("../images/", string(di), "/", string(ti), "/", string(ii), "/");
                files = dir(strcat(path, extension));
                for file = files'
                    ai = [];
                    ai = [ai SSN(file.name, ni)];
                    ai = [ai string(file.name)];
                    ai = [ai upper(ii)];
                    ai = [ai upper(ti)];
                    A = [A; ai];
                end
            end
        end
        output_name = strcat(string(di), "/", string(di), "_", string(ni), ".csv");
        writematrix(A, output_name);
    end
    strcat("OK: ", string(ni))
end
