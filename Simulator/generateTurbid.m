

path = 'Dataset/GroundTruth';
pngFiles = dir(strcat(path,'/*.jpg'));

pathOut = 'Dataset/Training';




l=3.5;              %Schechner,2006
T=1.0; %1.0; %Transmission coefficient at the water surface - from Processing of field spectroradiometric data - EARSeL
I0=1.0; %Cor da Luz - Branca Pura


initialDistance = 0.4;
finalDistance = 2;


profundity = 1.2;


distStep = 0.3;
distanceVec = initialDistance:distStep:finalDistance;
levels = length(distanceVec);




for  k = 1:length(pngFiles)
    filename = pngFiles(k).name;    
    I = imread(strcat(path,'/',filename));
   
    
    delta = 10;
    %Energy = zeros(size(input,1),size(input,2),size(400:delta:720));

    c = zeros(3,1);
    sumWeights = zeros(1,3);
    K=6;
    for wave = 400:delta:800
    
        load('deepgreen')
        % Get the c for this wavelenght
        cwave = feval(deepgreen,wave);



        % for rgb
        weights = spectrumRGB(wave)/(length(400:delta:800)/K);
    
     

        for i=1:3

            c(i) = c(i) + cwave * weights(i);

        end
    
    end  
        

    
    for i=1:3

        Binf(i)=l*T*I0*exp(-c(i)*double(profundity));

    end
    
    
    for i=1:levels
    
        distance = distanceVec(i);
        

        Timage = simulateTurbidImage(I,Binf,c,distance);

        
        imwrite(Timage,sprintf('%s/Green%d_%s',pathOut,i,filename));
        
    end
        
end
