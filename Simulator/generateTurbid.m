

path = 'Dataset/GroundTruth';
pngFiles = dir(strcat(path,'/*.jpg'));

pathOut = 'Dataset/Training';


for  k = 1:length(pngFiles)
    filename = pngFiles(k).name;    
    I = imread(strcat(path,'/',filename));
    
    
    
    Binf = [ 0.2, 0.6, 0.85];
    c = [ 4.0, 2.8 , 2.5];




    %[J, spImage] = spAverageImage(imvec{i} ,96);
    T =simulateTurbidImage(I,Binf,c,[]);
    
    imwrite(T,strcat(pathOut,'/Blue_',filename));
        
end
