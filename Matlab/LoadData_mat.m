%% LOAD AND VISUALIZE THE CLOUD POINTS FROM THE .mat FILES
addpath('Kin2-master');

% loop to visualize the cloud points
while true
    user = input('Enter the name of the file you want to visualize (press on "q" to exit): \n','s');
    
    if(user=='q') % control to exit from the loop
        break
    end

    folder = pwd;
    absoluteFileName = fullfile(folder, '/Dataset_Struct/', user);
    
    load(absoluteFileName); % read from file
    
    % images sizes
    c_width = 1920; c_height = 1080;
    
    % color stream figure
    c.h = figure; % set a picture window (frame, axes and image (color) with zero gray level thresholds)
    c.ax = axes;
    c.im = imshow(color,[]);
    
    model = zeros(3,1347); % set a zero matrix of 3x1347 
    figure, hmodel = plot3(model(1,:),model(2,:),model(3,:),'.');  % print - and save in hmodel - the newly created matrix on a 3D graphic (in the figure window) '.' is the marker symbol
    xlabel('X'), ylabel('Y'), zlabel('Z');
    
    k2 = Kin2('color','HDface');
    c.im = imshow(color, 'Parent', c.ax); % show color variable content, setting the axes
    
    % Display the HD faces data and face model(1347 points). Parameters:
    % 1) image axes
    % 2) faces structure obtained with getFaces
    % 3) display HD face model vertices(1347 points)?
    % 4) display text information (animation units)?
    % 5) text font size in pixels
    k2.drawHDFaces(c.ax,faces,true,true,20);
    
    model = faces(1).FaceModel; % save the model (3x1347) of the first face in faces in model variable
    set(hmodel,'XData',model(1,:),'YData',model(2,:),'ZData',model(3,:)); % add the data kept by model to 3D graphic previously created
    view(0,90)
    
end

k2.delete;