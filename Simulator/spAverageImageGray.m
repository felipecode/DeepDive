function [avgVector, spImage] = spAverageImageGray(image,sizeSps)



    %num=round((size(image,1)*size(image,2))/(sizeSps*sizeSps));
    % calculate the super pixels.
    imgtst=double(image);
    size(imgtst)
    imgtst = single(imgtst);
    spImage = vl_slic(imgtst, sizeSps, 1);

   
    % get how many superpixels you have
    
    
    
    nSps = max(max(spImage));
    %nChannels = size(image,3);


    avgVector = zeros(nSps,1);

    
   % avgImageR = zeros(size(image,1),size(image,2));
    
   % avgImageG = zeros(size(image,1),size(image,2));
    
   % avgImageB = zeros(size(image,1),size(image,2));
    
    for i=1:nSps
        
        mask = (spImage == i);
        nPixels = sum(sum(mask));
       
     
        avgVector(i) = sum(sum(imgtst.*double(mask)))/double(nPixels);
%        avgVector(i,2) = sum(sum(imgtst(:,:,2).*double(mask)))/double(nPixels);
%        avgVector(i,3) = sum(sum(imgtst(:,:,3).*double(mask)))/double(nPixels);
        
    end


end


