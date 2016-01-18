function [avgVector] = spAverageImageWSpImageGray(image,spImage)



    %num=round((size(image,1)*size(image,2))/(sizeSps*sizeSps));
    % calculate the super pixels.

%     size(imgtst)
     imgtst = single(image);
%     spImage = vl_slic(imgtst, sizeSps, 1);

   
    % get how many superpixels you have
    
    
    
    nSps = max(max(spImage));
    %nChannels = size(image,3);


    avgVector = zeros(nSps,1);

    
   % avgImageR = zeros(size(image,1),size(image,2));
    
   % avgImageG = zeros(size(image,1),size(image,2));
    
   % avgImageB = zeros(size(image,1),size(image,2));
    tic
    for i=1:nSps
        
        %mask = (spImage == i);
        
        imagepart = imgtst(spImage == i);
        nPixels = length(imagepart);
        %avgVector(i,1) = sum(sum(imgtst.*double(mask)))/double(nPixels);
        avgVector(i,1) = sum(sum(imagepart))/single(nPixels);

        
    end
    toc

end


