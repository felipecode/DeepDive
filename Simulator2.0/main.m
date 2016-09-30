%% TURBI SIMULATOR
% This is a model-based turbidity simulator to produce the degradation presented in real underwater images in 
% non-degraded images.
% This degradation can be according to the relation of the objects' distance or with a constant distance.
%
%% target config
%
datasetName = 'datasetNameDepth';
basePath = '/home/nautec/DeepDive-master/datasets/'; % must to be set where the dataset will be placed
patchSize = [];     % patch size; must be [] when not used (used when using sample patch)
numSample = 1;      % number of samples from an image (must be 1 when not using sample patch)
saveLevelDB = true;
loadFromMat = true; % must to be set as true (when loading from a .mat file) 
                    % or false (when load from image files)
%
%% source config
%
turbidityDatabasePath = 'TurbidityAzul';       % tubidity reference dataset
imageDatabasePath='GeraldepthDataset224.mat';  % base image dataset (can be image files or .mat)
%
%% Main
%
createFolders(basePath,datasetName);                        % create folder structure
turbidityImages = loadImages(turbidityDatabasePath,'png');  % load images from tubidity database
if loadFromMat
    loadedFile=load(imageDatabasePath);
    baseImages = loadedFile.('images');
    distances = loadedFile.('depths');
else
    baseImages = loadImages(imageDatabasePath,'jpg');       % load base images from image database folder; Set the file extension
end
W = waitbar(0,'Go drink a coffe, it will take a while...'); % Wait bar
steps=numSample*length(baseImages)*length(turbidityImages);
%
for k = 1:numSample
    for i = 1: length(baseImages)
        
        % base images and distances must be double!!! Check it
        if loadFromMat
            input=double(baseImages(:,:,:,i))/255.0;        % load images and depths from .mat file;    
            distance=distances(:,:,i);
        else
            input = double(baseImages{i})/255.0;
        end
        if not(isempty(patchSize))
            input = samplePatch(input,patchSize(1),patchSize(2)); % It take a sample from input using patch size selected
        end
        
        for j=1:length(turbidityImages)
             turbidityPatch = double(turbidityImages{j})/255.0;
             if not(exist('distance','var'))
                distance = rand()*ones(size(input(:,:,1))); % if the distance is not defined by the base images it will be 
                                                            % generate a random and constant distance
             end
             [turbidImage,transmission]=simulateTurbidity(input,turbidityPatch,distance); 
             turbidImageIndex=((k-1)*length(baseImages)+i-1)*length(turbidityImages)+j;
             waitbar(turbidImageIndex/steps);
        if rem(turbidImageIndex,300)==0
            save('atual.mat','k','i','j');
        end
        saveTurbidImage(turbidImage,input,transmission,basePath,datasetName,i,turbidImageIndex)     % save image based on index
        end
    end
end
close(W) 