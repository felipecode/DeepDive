% Compute the psf as described by the paper " A new convolution kernel
%for atmospheric point spread function"

function psfKernel = psf(sizeKernel,g,c)

    
    psfKernel = zeros(sizeKernel*2 + 1);
    k=7;
    c = c*k;
    
    for i=0:sizeKernel
        
        for j=0:sizeKernel
        
            % Put into metric scale
            x = i*0.0002;
            y = j*0.0002;
            upperFracExp = (x^2 + y^2)^(c/2);
            lowerFracExp = abs(calcA(c,(1-g)/g))^c;
            upperFrac = exp(-(upperFracExp/lowerFracExp));
            lowerFrac = 4*(gamma(1+(1/c))^2)*(calcA(c,(1-g)/g)^2);
            
            psfKernel(i+sizeKernel+1,j+sizeKernel+1) = upperFrac/lowerFrac;
            psfKernel(sizeKernel+1 - i, sizeKernel+1 +j) = upperFrac/lowerFrac;
            psfKernel(sizeKernel+1 - i, sizeKernel+1 -j) = upperFrac/lowerFrac;
            psfKernel(sizeKernel+1 + i, sizeKernel+1 -j) = upperFrac/lowerFrac;
        end
        
    end
    psfKernel = psfKernel./sum(sum(psfKernel));
        

end

function a = calcA(p,sigma)


   a = sqrt((sigma^2)*((gamma(1/p))/(gamma(3/p))));

end