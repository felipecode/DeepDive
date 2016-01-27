function Ed = directComponent(J, cmap,dmap)

% It may be interesting to use some particle oscilation to simulate.


photonDensity = 100;

[sizeX sizeY m] = size(J)


Ed = zeros(size(J));

for i=1:sizeX
    for j=1:sizeY
        
        Intensity = J(i,j);
        nPhotons = photonDensity*Intensity;
        count = 0; % This counts the number of photons 
        P = exp(-cmap(i,j)*dmap(i,j));
        for k=1:nPhotons
            
            if (double(randi(100000))/100000) < P
                count = count +1;
                
            end
                
            
        end
        
        Ed(i,j)=count;
    end

end
Ed = double(Ed)/255;




