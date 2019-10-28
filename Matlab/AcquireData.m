%% DATA ACQUISITION IN THE .mat FORMAT FILE
% Storage of the whole structured array obtained by getHDFaces 
clear all
close all

addpath('Kin2-master');
addpath('Kin2-master/Mex');

% Create Kinect 2 object and initialize it
% Available sources: 'color', 'depth', 'infrared', 'body_index', 'body', 'face' and 'HDface'
k2 = Kin2('color','HDface');

% images sizes
c_width = 1920; c_height = 1080;

% Color image is too big, let's scale it down
COL_SCALE = 1.0;

% Create matrices for the images
color = zeros(c_height*COL_SCALE,c_width*COL_SCALE,3,'uint8');

% color stream figure
c.h = figure; % set a picture window (frame, axes and image (color) with zero gray level thresholds)
c.ax = axes;
c.im = imshow(color,[]);
title('Color Source (press q to exit)');
set(gcf,'keypress','k=get(gcf,''currentchar'');'); % listen keypress

model = zeros(3,1347); % set a zero matrix of 3x1347 
figure, hmodel = plot3(model(1,:),model(2,:),model(3,:),'.');  % print - and save in hmodel - the newly created matrix on a 3D graphic (in the figure window) '.' is the marker symbol
title('HD Face Model (press q to exit)')
xlabel('X'), ylabel('Y'), zlabel('Z');
set(gcf,'keypress','k=get(gcf,''currentchar'');'); % listen keypress

% Loop until pressing 'q' on any figure
k=[];
expr = 0;
count = 0;
user = input('Enter your username code: \n','s');
while true
    % Get frames from Kinect and save them on underlying buffer
    validData = k2.updateData;
    
    % Before processing the data, we need to make sure that a valid frame was acquired.
    if validData % this function gets data from the Kinect and save it in an internal buffer. If there is a valid frame, returns 1 if not, returns 0.
        % Get color frame
        color = k2.getColor; % Save in color variable a 1920 x 1080 3-channel color frame frame from Kinect V2. 
        
        % update color figure
        color = imresize(color,COL_SCALE);
        c.im = imshow(color, 'Parent', c.ax); % show color variable content, setting the axes
        
        % Get the HDfaces data the output faces is a structure array with at most 6 faces. Each face has the following fields:
        % - FaceBox: rectangle coordinates representing the face position in color space. [left, top, right, bottom].        
        % - FaceRotation: 1 x 3 vector containing: pitch, yaw, roll angles
        % - HeadPivot: 1 x 3 vector, computed center of the head, which the face may be rotated around. This point is defined in the Kinect body coordinate system. 
        % - AnimationUnits: 17 animation units (AUs). Most of the AUs areexpressed as a numeric weight varying between 0 and 1.
        % - ShapeUnits: 94 hape units (SUs). Each SU is expressed as a numeric weight that typically varies between -2 and +2.
        % - FaceModel: 3 x 1347 points of a 3D face model computed by face capture
        faces = k2.getHDFaces('WithVertices','true');                    

        % Display the HD faces data and face model(1347 points). Parameters: 
        % 1) image axes
        % 2) faces structure obtained with getFaces
        % 3) display HD face model vertices (1347 points)
        % 4) display text information (animation units)
        % 5) text font size in pixels
        k2.drawHDFaces(c.ax,faces,true,true,20);
             
        % Plot face model points
        if size(faces,2) > 0 % faces is an array of objects, with 2 returns the number of columns
             model = faces(1).FaceModel; % save the model (3x1347) of the first face in faces in model variable
             set(hmodel,'XData',model(1,:),'YData',model(2,:),'ZData',model(3,:)); % aggiunge al grafico 3D creato in precedenza i dati contenuti in model
             view(0,90)
         end
    end
    
    % If user presses 'q', exit loop
    if ~isempty(k)
        if strcmp(k, 'q')
            fprintf('The end\n');
            break;
        end
        if strcmp(k, '1')
            expr='felice';
            count = 0;
            fprintf('Happy\n');
            k=[];
        end
        if strcmp(k, '2')
            expr='triste';
            fprintf('Sad\n');
            count = 0;
            k=[];
        end
        if strcmp(k, '3')
            expr='spaventato';
            fprintf('Scared\n');
            count = 0;
            k=[];
        end
        if strcmp(k, '4')
            expr='arrabbiato';
            fprintf('Angry\n');
            count = 0;
            k=[];
        end
        if strcmp(k, '5')
            expr='disgustato';
            fprintf('Disgusted\n');
            count = 0;
            k=[];
        end
         if strcmp(k, '6')
            expr='sorpreso';
            fprintf('Surprised\n');
            count = 0;
            k=[];
        end
        if strcmp(k, 'c')
            chr = int2str(count);
            folder = pwd;
            path = fullfile(folder, '/Dataset_Struct/');
            save(strcat(path, user, '_', expr, '_', chr), 'faces', 'color'); 
            count=count+1;
            fprintf('The photo has been taken\n');
            k=[];
        end    
    end
  
    pause(0.02)
end

% Close kinect object
k2.delete;

%close all;
