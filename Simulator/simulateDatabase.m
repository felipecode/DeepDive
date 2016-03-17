clear


databaseName = 'Dataset2_0';
numberOfImages = 200000;
pathImageDatabase='ImageDataBase';
pathTurbidityDatabase = 'TurbidityDatabase';


% After this level of turbidity we ignore.
maxLevel =6;


% Number of the images that are going to be loaded each time for sampling.
nLoadedImages =687;

turbidityImages = loadImages(pathTurbidityDatabase,'png');


distanceVec = 0.1:0.1:10;
noiseVec = 0.001:0.001:0.02;
patchSizeX = 128;
patchSizeY = 128;


mkdir(sprintf('%s/Training',databaseName));

mkdir(sprintf('%s/Validation',databaseName)); 

mkdir(sprintf('%s/Test',databaseName));

mkdir(sprintf('%s/TestGroundTruth',databaseName)); 

mkdir(sprintf('%s/GroundTruth',databaseName)); 

mkdir(sprintf('%s/ValidationGroundTruth',databaseName)); 

for i=0:maxLevel
  
    mkdir(sprintf('%s/Training/%d',databaseName,i)); 
    mkdir(sprintf('%s/Validation/%d',databaseName,i)); 
    mkdir(sprintf('%s/GroundTruth/%d',databaseName,i)); 
    mkdir(sprintf('%s/ValidationGroundTruth/%d',databaseName,i)); 
    mkdir(sprintf('%s/Test/%d',databaseName,i)); 
    mkdir(sprintf('%s/TestGroundTruth/%d',databaseName,i)); 
end



images = loadImages(pathImageDatabase,'jpg',1,nLoadedImages);
i=1;
while i <= numberOfImages

    % Check memory issues when loading images.
    % May Sample
    if rem(i,500)==0
        i
    end
   % if rem(i,nLoadedImages) == 1
   %     images = loadImages(pathImageDatabase,'jpg',i,nLoadedImages);
   % end
    
    %%% Try a random sample without reposition. 
    
    position = randsample(1:length(images),1,true);
    image = images{position};
    %images(position) = [];

    
    % Samples a turbidity patch to be used.
    position = randsample(1:length(turbidityImages),1,true);
    turbidPatch = turbidityImages{position};


    
    
                
    % Sample the distance from a distance vec 
    
    %distance  = randsample(distanceVec,1);
    distance = sampleDistance(1,100);
    
    % Sample a noise value to be used on the function 
    
    noise =randsample(noiseVec,1);
    
    
    patch = samplePatch(image,patchSizeX,patchSizeY);
    
    % Sample noise from a function
    
    
    % Sample forward scattering from a function.
    

    forward = sampleForwardScattering(1.7,50);
    
    [output,groundTruth] = applyTurbidity(patch,turbidPatch,forward,distance,noise);
    
    % Check the level of turbidity.
    
    
    vifR = vifvec(double(patch(:,:,1))./255,output(:,:,1));
    vifG = vifvec(double(patch(:,:,2))./255,output(:,:,2));
    vifB = vifvec(double(patch(:,:,3))./255,output(:,:,3));

    vif = (log(vifR)+log(vifG)+log(vifB))/3;

    % select 10 percent to be used on validation.
    lotery = randi([1 10],1,1);
    
    if uint8(round(-(vif))) > maxLevel
        %'cont'
        continue;
    end
    
    if lotery == 10
        % Put the data into the validation set
        imwrite(output,sprintf('%s/Validation/%d/%d.jpg',databaseName,min(9,uint8(round(-(vif)))),i));
        imwrite(groundTruth,sprintf('%s/ValidationGroundTruth/%d/%d.jpg',databaseName,min(9,uint8(round(-(vif)))),i));
    
    elseif lotery == 9
        % Put it on test set
        imwrite(output,sprintf('%s/Test/%d/%d.jpg',databaseName,min(9,uint8(round(-(vif)))),i));
        imwrite(groundTruth,sprintf('%s/TestGroundTruth/%d/%d.jpg',databaseName,min(9,uint8(round(-(vif)))),i));    
    else
        imwrite(output,sprintf('%s/Training/%d/%d.jpg',databaseName,min(9,uint8(round(-(vif)))),i));
        imwrite(groundTruth,sprintf('%s/GroundTruth/%d/%d.jpg',databaseName,min(9,uint8(round(-(vif)))),i));    
    end
    
    i= i+1;

end