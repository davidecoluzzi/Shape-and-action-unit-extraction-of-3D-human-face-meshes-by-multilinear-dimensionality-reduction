%% CONVERSION OF THE CLOUD POINTS IN THE .csv FORMAT FILE
countU = 1;
espr = 'felice';
count = 0;
% loop to read .mat files and save .csv files
while true
    if(count==4)
        espr = 'triste';
    end
    if(count==8)
        espr = 'spaventato';
    end
    if(count==12)
        espr = 'arrabbiato';
    end
    if(count==16)
        espr = 'disgustato';
    end
    if(count==20)
        espr = 'sorpreso';
    end
    if(count==24)
        countU=countU+1;
        espr = 'felice';
        count=0;
    end
    
    countF = mod(count, 4);
    
    countu = int2str(countU); % convert to char
    countf = int2str(countF); % convert to char
    user = strcat('u', countu, '_', espr, '_', countf); % concatenate to create the name
    
    try
        folder = pwd;
        absoluteFileName = fullfile(folder, '/Dataset_PointModel/', user);
        load(absoluteFileName);
        absoluteFileName2 = fullfile(folder, '/Dataset_Csv/', user);
        path = strcat(absoluteFileName2, '.csv');
        csvwrite(path, pointModel); % save in .csv format file
    catch
    end
    
    count = count + 1;
    
    if(countU==22) % break the loop
        break;
    end
end