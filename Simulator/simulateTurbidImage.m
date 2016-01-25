function T = simulateTurbidImage(input,Binf,c,distance)

    % Set distance of the scene, for our experiment is always 0.58
    

     averageCossine = 0.75;


     TR = calculateTurbidImageFixD(double(input(:,:,1))/255,c(1),distance,averageCossine,Binf(1)); 
     averageCossine = 0.714;
     
     
     TG = calculateTurbidImageFixD(double(input(:,:,2))/255,c(2),distance,averageCossine,Binf(2)); 
%     
     averageCossine = 0.681;
     TB = calculateTurbidImageFixD(double(input(:,:,3))/255,c(3),distance,averageCossine,Binf(3)); 
%     
%     
     T(:,:,1) = TR;
     T(:,:,2) = TG;
     T(:,:,3) = TB;
     

end
