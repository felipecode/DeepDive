  
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
input = imread('UWSIM1_17.jpg');
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






input = imresize(input,size(dmapOutput));

%[J, spImage] = spAverageImage(imvec{i} ,96);
% delta = 10;
% %Energy = zeros(size(input,1),size(input,2),size(400:delta:720));
% 
% c = zeros(3,1);
% sumWeights = zeros(1,3);
% K=6;
% for wave = 400:delta:800
%     % Load an absorption model of a certain water tipe
%     load('TurbidityDatabase/deepgreen')
%     % Get the c for this wavelenght
%     cwave = feval(mediafunction,wave);
%     
% 
%    
%     % for rgb
%      weights = spectrumRGB(wave)/(length(400:delta:800)/K);
%     
%      
% 
%     for i=1:3
%                
%         c(i) = c(i) + cwave * weights(i);
%     
%     end
%     
%     %sumWeights = sumWeights + weights;
% 
% 
% end





l=1.06;              %Schechner,2006
T=1; %1.0; %Transmission coefficient at the water surface - from Processing of field spectroradiometric data - EARSeL
I0=1.0; %Cor da Luz - Branca Pura

profundity = 1;

patch = double(imread('TurbidityDatabase/mujereswater.png'))/255;
c = acquireWaterProperties(patch);
for i=1:3

    Binf(i)=l*T*I0*exp(-c(i)*double(profundity));


end

distance =5;

%fwcossine = 0.1*[ 0.75 0.714 0.681];
fwintesity = 50;
T = simulateTurbidImage(input,Binf,c,distance,0.01,fwintesity);
GT  = simulateTurbidImageGT(input,c,distance,3);    

figure; imshow(T);

%figure; imshow(GT);
