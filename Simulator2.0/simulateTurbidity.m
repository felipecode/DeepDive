function [turbidPatch,trans] = simulateTurbidity(patch,turbidPatch,distance)
    [c,Binf] = acquireProperties(turbidPatch);
    
    J=patch;
    trans=ones([size(patch)]);
    for i=1:3
        trans(:,:,i)=exp(-distance*c(i));
    end
    Ed = J.*trans;                                      % Direct component
    for i=1:3
        Eb(:,:,i) = (Binf(i) - Binf(i).*trans(:,:,i));  % Backscattering component
    end
    turbidPatch = Ed + Eb;   
end