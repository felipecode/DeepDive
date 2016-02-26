function c=acquireWaterProperties(patch)

l=1.06;              %Schechner,2006
T=1; %1.0; %Transmission coefficient at the water surface - from Processing of field spectroradiometric data - EARSeL
I0=1.0; %Cor da Luz - Branca Pura

profundity = 1;

    
c = zeros(1,3);

% For each possible channel
for i=1:3
    patchChannel = patch(:,:,i);
    
    patchVec = patchChannel(:);

    for k=1:length(patchVec)
        pixel = max(patchVec(k),0.001);
        
        cFun = -log(pixel/(l*T*I0))/profundity; 
        c(i) = c(i) + cFun;
        
        %pixel
    end
    c(i) = c(i)/length(patchVec);
    
end



end