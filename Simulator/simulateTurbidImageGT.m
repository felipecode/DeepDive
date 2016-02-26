function T = simulateTurbidImageGT(input,c,distance,minpixel)

    % Set distance of the scene, for our experiment is always 0.58
    

     TR = calculateGTImageFixD(double(input(:,:,1)),c(1),distance,minpixel); 

     
     
     TG = calculateGTImageFixD(double(input(:,:,2)),c(2),distance,minpixel); 

     TB = calculateGTImageFixD(double(input(:,:,3)),c(3),distance,minpixel); 
%     
%     
     T(:,:,1) = TR;
     T(:,:,2) = TG;
     T(:,:,3) = TB;
     

end
