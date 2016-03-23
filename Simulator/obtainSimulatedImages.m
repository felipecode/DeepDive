
pathImageDatabase='ImageDataBase';
pathTurbidityDatabase = 'TurbidityDatabase';


% After this level of turbidity we ignore.



% Number of the images that are going to be loaded each time for sampling.
nLoadedImages =685;

turbidityImages = loadImages(pathTurbidityDatabase,'png');


distanceVec = 0.1:0.1:10;
noiseVec = 0.001:0.001:0.02;
patchSizeX = 512;
patchSizeY = 512;





i=1;
%images
%images = dir(strcat(pathImageDatabase,'/*.','jpg'));
images = loadImages(pathImageDatabase,'jpg',i,nLoadedImages);

while i <= numberSimulatedImages
    
    % Check memory issues when loading images.
    % May Sample
    if rem(i,500)==0
        i
    end
   % if rem(i,nLoadedImages) == 1
   %     images = loadImages(pathImageDatabase,'jpg',i,nLoadedImages);
   % end
    
    %%% Try a random sample with reposition. 
    
    position = randsample(1:length(images),1,true);
    %position =i;
    %image = imread(strcat('ImageDataBase/',images(position).name));
    image = images{position};
    %images(position) = [];

    
  %  [sizeX,sizeY,sizeZ]  = size(image);
    % remove overlly sized images
%     if sizeX >512 || sizeY>512
%           
%         delete(strcat('ImageDatabaseNoWater/',images(position).name));
%         continue;
%         
%     end
   
    
    
    
    if size(image,3) ==1
       
        imageaux = zeros(size(image,1),size(image,2),3);
        imageaux(:,:,1) = image;
        imageaux(:,:,2) = image;
        imageaux(:,:,3) = image;
        image = imageaux;
        
    end
    
    
   
    % Samples a turbidity patch to be used.
    position = randsample(1:length(turbidityImages),1,true);
    turbidPatch = turbidityImages{position};


    if rem(randi(2),2)==0
       
        scale = randsample([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],1);
        
    else
        scale=1;
       
    end
    
                
    % Sample the distance from a distance vec 
    
    %distance  = randsample(distanceVec,1);
    distance = sampleDistance(0.6,100);
    
    % Sample a noise value to be used on the function 
    
    noise =randsample(noiseVec,1);
    
    
    patch = samplePatch(image,patchSizeX,patchSizeY);
    %patch = image;

    
    % apply a scale to the patch ( DATA AUGMENTATION )
    
    patch = imresize(patch,scale,'nearest');
    patch = imresize(patch,1/scale,'nearest');
    
    
    % Sample forward scattering from a function.
    

    forward = sampleForwardScattering(1.7,50);
    
    [output,groundTruth] = applyTurbidity(patch,turbidPatch,forward,distance,noise);
    
    % Check the level of turbidity.
    
    
%     vifR = vifvec(double(patch(:,:,1))./255,output(:,:,1));
%     vifG = vifvec(double(patch(:,:,2))./255,output(:,:,2));
%     vifB = vifvec(double(patch(:,:,3))./255,output(:,:,3));
% 
%     vif = (log(vifR)+log(vifG)+log(vifB))/3;
% 
%     select 10 percent to be used on validation.
     lotery = randi([1 10],1,1);
%     
%     if uint8(round(-(vif))) > maxLevel
%         'cont'
%         continue;
%     end


    [sizeX,sizeY,sizeZ]  = size(output);
    
    imaux = zeros(512,512,3);
    imaux(1:sizeX,1:sizeY,:) = output;
    output = imaux;
    imaux = zeros(512,512,3);
    imaux(1:sizeX,1:sizeY,:) = groundTruth;
    groundTruth = imaux;
    
    


    if lotery == 10
%        Put the data into the validation set
        imwrite(output,sprintf('%s/Validation/%d.jpg',databaseName,i));
        imwrite(groundTruth,sprintf('%s/ValidationGroundTruth/%d.jpg',databaseName,i));
    
    elseif lotery == 9
        %Put it on test set
        imwrite(output,sprintf('%s/Test/%d.jpg',databaseName,i));
        imwrite(groundTruth,sprintf('%s/TestGroundTruth/%d.jpg',databaseName,i));    
    else
        imwrite(output,sprintf('%s/Training/%d.jpg',databaseName,i));
        imwrite(groundTruth,sprintf('%s/GroundTruth/%d.jpg',databaseName,i));    
    end
    
    i= i+1;

end