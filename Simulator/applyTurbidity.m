function [turbidPatch,groundTruth] = applyTurbidity(patch,turbidPatch,forward,distance,noise)


patch = double(patch)./255;
turbidPatch = double(turbidPatch)./255;

l=1.06;              %Schechner,2006
T=1; %1.0; %Transmission coefficient at the water surface - from Processing of field spectroradiometric data - EARSeL
I0=1.0; %Cor da Luz - Branca Pura

profundity = 1;


c = acquireWaterProperties(turbidPatch);

for i=1:3

    Binf(i)=l*T*I0*exp(-c(i)*double(profundity));


end


%fwcossine = 0.1*[ 0.75 0.714 0.681];

turbidPatch = simulateTurbidImage(patch,Binf,c,distance,noise,forward);
groundTruth  = simulateTurbidImageGT(patch,c,distance,2);    



end