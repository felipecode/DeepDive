function imvec = loadImages(path,format,initial,nImages)
    

    pngFiles = dir(strcat(path,'/*.',format));

    if (nargin<3)
        nImages=length(pngFiles);
        initial =1;
    end;

    


    %imvec = zeros(size()
    imvec{nImages} = 1;
    for k = initial:(initial+nImages-1)
       
        filename = pngFiles(k).name;
        [I, MAP] = imread(strcat(path,'/',filename));
        
        %H = fspecial('gaussian',[round(window*sigma), round(window*sigma)],sigma);
        %Ilarge = uint8(zeros(size(I,1)+1,size(I,2)+2));
        
        %Ilarge(1:size(I,1),1:size(I,2))  = I;

        % Convert indexed image to true-color (RGB) format
%       
%         RGB=I;
%         % Convert image to L*a*b* color space
%         cform2lab = makecform('srgb2lab');
%         LAB = applycform(RGB, cform2lab); 
% 
%         % Scale values to range from 0 to 1
%         L = LAB(:,:,1);%/100; 
% 
%         % Perform CLAHE
%         LAB(:,:,1) = adapthisteq(L,'NumTiles',...
%                                  [8 8],'ClipLimit',0.0005);%*100;
% 
%         % Convert back to RGB color space
%         cform2srgb = makecform('lab2srgb');
%         J = applycform(LAB, cform2srgb); 

        % Display the results
        %figure, imshow(RGB); 
        %figure, imshow(J);

%         
%         J=I;
%         %Ilarge = clahs(Ilarge,7,4,'cliplimit',3,'PreserveMaxMin');
%         %I = Ilarge(1:size(I,1),1:size(I,2));
%         %figure; imshow(I);
%         GaussBlur = imfilter(J(:,:,1),H,'same');
%         J(:,:,1) = GaussBlur;
%         GaussBlur = imfilter(J(:,:,2),H,'same');
%         J(:,:,2) = GaussBlur;
%         GaussBlur = imfilter(J(:,:,3),H,'same');
%         J(:,:,3) = GaussBlur;
        
        %I = imadjust(I);
        %figure; imshow(GaussBlur);
        imvec{k-initial+1} = I;

    end

end


