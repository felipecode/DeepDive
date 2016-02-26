function T = simulateTurbidImage(input,Binf,c,distance,noise,fwintensity)

    % Set distance of the scene, for our experiment is always 0.58
    
    fwcossine = [ 0.75 0.714 0.681];
     


     TR = calculateTurbidImageFixD(double(input(:,:,1)),c(1),distance,fwcossine(1),Binf(1),noise,fwintensity); 

     
     
     TG = calculateTurbidImageFixD(double(input(:,:,2)),c(2),distance,fwcossine(2),Binf(2),noise,fwintensity); 
%     

     TB = calculateTurbidImageFixD(double(input(:,:,3)),c(3),distance,fwcossine(3),Binf(3),noise,fwintensity); 
%     
%     
     T(:,:,1) = TR;
     T(:,:,2) = TG;
     T(:,:,3) = TB;
     

end
