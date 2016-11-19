%% TURBI SIMULATOR
% This is a model-based turbidity simulator to produce the degradation presented in real underwater images in 
% non-degraded images.
% This degradation can be according to the relation of the objects' distance or with a constant distance.
%
%% target config
%
datasetName = 'datasetNameDepth';
basePath = '../datasets/simulator_data'; % must to be set where the dataset will be placed
patchSize = [];     % patch size; must be [] when not used (used when using sample patch)
numSample = 1;      % number of samples from an image (must be 1 when not using sample patch)
saveLevelDB = true;
loadFromMat = true; % must to be set as true (when loading from a .mat file) 
                    % or false (when load from image files)
%
%% source config
%
imageDatabasePath='../datasets/YUVdatasetDepth224.mat';  % base image dataset (can be image files or .mat)
%
%% Main
%
imgs_dir=strcat(basePath,'/images');
dists_dir=strcat(basePath,'/depths');
mkdir(imgs_dir);
mkdir(dists_dir);
%createFolders(basePath,datasetName);                        % create folder structure
turbidityImages = loadImages(turbidityDatabasePath,'png');  % load images from tubidity database
if loadFromMat
    loadedFile=load(imageDatabasePath);
    baseImages = loadedFile.('images');
    distances = loadedFile.('depths');
else
    baseImages = loadImages(imageDatabasePath,'jpg');       % load base images from image database folder; Set the file extension
end

n_images=length(baseImages);

for i = 1: n_images
	input=baseImages(:,:,:,i);        % load images and depths from .mat file;    
	distance=distances(:,:,i);
	filename=strcat(num2str(i),'.png');
	imwrite(input,strcat(imgs_dir,strcat('/',filename)));
	distance_norm=distance/10;
	imwrite(distance_norm,strcat(dists_dir,strcat('/',filename)));
end
