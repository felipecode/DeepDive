function imvec = loadImages(path,format)
   
    imgNames = dir(strcat(path,'/*.',format));
    nImages=length(imgNames);
    imvec{nImages} = 1;
    
    for k = 1:nImages
        filename = imgNames(k).name;
        I = imread(strcat(path,'/',filename));
        imvec{k} = I;
    end
end

