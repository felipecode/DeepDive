function [c, Binf]=acquireProperties(patch)     %it estimates c from a given turbidity patch
    l=1.06;      %Schechner,2006
    T=1;         %Transmission coefficient at the water surface - from Processing of field spectroradiometric data - EARSeL
    I0=1.0;      %Cor da Luz - Branca Pura
    patch = max(patch,0.001);
    patch = -log(patch/(l*T*I0));
    c= mean(mean(patch));
    for i=1:3
        Binf(i)=l*T*I0*exp(-c(i)*double(max(max(patch(:,:,i)))));   
    end
end