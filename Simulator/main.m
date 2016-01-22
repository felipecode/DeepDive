  
% Color Hard Green

% Binf = [ 0.2, 0.85, 0.4];
% c = [ 4.0, 2.3 , 2.8];


% Color Hard Blue

% Binf = [ 0.2, 0.6, 0.85];
% c = [ 4.0, 2.8 , 2.5];


% Color Ciano Blue


%Binf = [ 0.2, 0.85, 0.85];
%c = [ 4.0, 2.8 , 2.8];


%image = imvec{18};
input = imvec{1};
%distance = 0.58;
%dmapR = calculateDmap(image(:,:,1),distance); 


% [I, spImage] = spAverageImage(image,164);
% dmapR = calculateDmap(image(:,:,1),distance); 
  
% [dmapInput] = spAverageImageWSpImageGray(dmapR,spImage);
dmapOutput = imread('dmap.png');
dmapOutput = double(dmapOutput)/255;


% input = imvec{1};
% [J] = spAverageImageWSpImage(I1, spImage);
 


Binf = [ 0.2, 0.85, 0.85];



c = [ 4.0, 2.8 , 2.8];


input = imresize(input,size(dmapOutput));

%[J, spImage] = spAverageImage(imvec{i} ,96);
T =simulateTurbidImage(input,Binf,c,dmapOutput);

figure; imshow(T);
