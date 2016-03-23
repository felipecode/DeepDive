


for i=1:numberRealImages
    % First we get the path of all images to be loaded.

    jpgFiles = dir(strcat('RealImagesDataBase','/*.','jpg'));

    % sample a random image to be used

    position = randsample(1:length(jpgFiles),1,true);

    % load the image 
    image = imread(strcat('RealImagesDataBase/',jpgFiles(position).name));
    
    
    % load the ground truth image
    
    groundtruth = imread(strcat('RealImagesDataBase/',jpgFiles(position).name));

    
    patch = samplePatch(image,patchSizeX,patchSizeY);
    
    % take this same patch from the ground truth
    
    
end





