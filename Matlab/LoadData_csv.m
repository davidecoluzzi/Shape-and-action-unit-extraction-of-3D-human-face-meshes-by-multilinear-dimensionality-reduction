%% LOAD AND VISUALIZE THE CLOUD POINTS FROM THE .csv FILES
addpath('Kin2-master');

% loop to visualize the cloud points
while true
    user = input('Enter the name of the file you want to visualize (press on "q" to exit): \n','s');
    
    if(user=='q') % control to exit from the loop
        break
    end

    folder = pwd;
    absoluteFileName = fullfile(folder, '/Dataset_Csv/', user);
    
    pointModel = csvread(absoluteFileName); % read from file
    
    model = zeros(3,1347); % set a zero matrix of 3x1347
    figure, hmodel = plot3(model(1,:),model(2,:),model(3,:),'.');  % print - and save in hmodel - the newly created matrix on a 3D graphic (in the figure window) '.' is the marker symbol
    xlabel('X'), ylabel('Y'), zlabel('Z');
    model = pointModel;
    
    set(hmodel,'XData',model(1,:),'YData',model(2,:),'ZData',model(3,:)); % add the data kept by model to 3D graphic previously created
    view(0,90)
    
end

k2.delete;