function idt = estimateIDT(J,I,Binf,dmap,i,spImage)

    % Set distance of the scene, for our experiment is always 0.58
    
    
    % Take just the luminance
    %[H S I ] = rgb2hsv(I);
    %[H S J ] = rgb2hsv(J);
    %Binf = double(rgb2gray(Binf))/255;    
    Binf = double(Binf);
    J = double(J);
    I = double(I);

    

   % Binf estimation could not be done for now.
   
    % = sort(Binf(:,1))
    %BinfG = sort(Binf(:,2))
    %BinfB = sort(Binf(:,3))
    
    
        
    %Bvec = sort(Binf,'descend');  

    %Binf = mean(Bvec(1:15))
        
  %  Bvec = sort(Binf(:,2),'descend');  

  %  BinfG = mean(Bvec(1:15))
        
   % Bvec = sort(Binf(:,3),'descend'); 

 %   BinfB = mean(Bvec(1:15))


    
    
    % Still need to regulate
    
    % Estimate turbid noise


        
    %cR = solveCfun (double(J(:,:,1)),Binf(:,1),I(:,1),0.78,dmap,spImage)
    
    
    idt = estimateC(J,I,dmap,Binf);
    fprintf('idt = %f',idt);
    
    
     %print(sprintf('%d_r',i),'-dpng');
     %close;
    %cG = solveCfun (double(J(:,:,2)),Binf(:,2),I(:,2),0.78,dmap,spImage)
   % cG = estimateC(J(:,2),I(:,2),dmap,BinfG);
   % fprintf('cG = %f',cG);
     %    print(sprintf('%d_g',i),'-dpng');
     %close;
    %cB = solveCfun (double(J(:,:,3)),Binf(:,3),I(:,3),0.78,dmap,spImage)
  %  cB = estimateC(J(:,3),I(:,3),dmap,BinfB);
 %   fprintf('cB = %f',cB);
     %    print(sprintf('%d_b',i),'-dpng');
     %close;
    


 
    
%     
     averageCossine = 0.75;
%     calculate the image with just the turbid noise
%     
     TR = calculateTurbidImage(J(:,:,1),cR,dmap,averageCossine,BinfR); 
     averageCossine = 0.714;
     
     
     TG = calculateTurbidImage(J(:,:,2),cG,dmap,averageCossine,BinfG); 
%     
     averageCossine = 0.681;
     TB = calculateTurbidImage(J(:,:,3),cB,dmap,averageCossine,BinfB); 
%     
%     
     T(:,:,1) = TR;
     T(:,:,2) = TG;
     T(:,:,3) = TB;
     
    % imwrite(T,sprintf('%d_Timage',i),'png');
     
%     
%     idtR = 100 * floor((1- ssim_index(J(:,:,1)/max(max(J(:,:,1))),TR))*10000)/10000
%     
%     idtG = 100 * floor((1- ssim_index(J(:,:,2)/max(max(J(:,:,2))),TG))*10000)/10000
%     idtB = 100 * floor((1- ssim_index(J(:,:,3)/max(max(J(:,:,3))),TB))*10000)/10000
%     
%     idtR = calculateError(J(:,:,1),TR,'MSE')
%     idtG = calculateError(J(:,:,2),TG,'MSE')
%     idtB = calculateError(J(:,:,3),TB,'MSE')
%     
%     
%     idt = (idtR + idtG + idtB)/2;
%idt=(cR + cG + cB)/3;
    
end
