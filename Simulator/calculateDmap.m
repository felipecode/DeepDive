function dmap =calculateDmap(image,z)

    [sx sy c] =size(image);
    centerx = round(sx/2);
    centery = round(sy/2);
    
    dmap = zeros(size(image(:,:,1)));
    for i=1:sx
        for j=1:sy
       
            x = abs(i - centerx)*0.0002;
            y = abs(j - centery)*0.0002;
            dmap(i,j) = sqrt(x^2 + y^2 + z^2);
            
        end
        
    end


end