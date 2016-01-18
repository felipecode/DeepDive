function c =estimateC(J,I,dmap,Binf)


    % TODO: add the forward scattering to this equation
   
    

    %4 pixels por mm 4000 per meter. It goes to the exponential as ln of
    %4000
    size(I)
    size(J)
    %size(Binf)
    % Operate the equation

    %Ed = J.*exp(-distance*c_mat);
    
    %Jf = conv2(Ed,psf(10,averageCossine,c),'same');
    
    
    %xi = 1203;
    %xf = xi+50;
 

    xi= 1;
    xf = size(I,1);
    
   % yi=1;
   % yf = size(I,2);
  
   % yi= 1500;
   % yf = yi+50;
%     tmap = zeros(size(I));
%     for i=1:size(I,1)
%         for j=1:size(J,2)
%             B = I(i,j) - Binf(i,j);
%             
%             C = J(i,j) - Binf(i,j);
%             
%             if C == 0
%                 C = 0.0000001;
%             end
%             
%             if B/C < 0.000000001
%                 tmap(i,j) = 0.00001;
%             elseif  B/C > 500
%                 tmap(i,j) = 500;
%             
%             else
%                 tmap(i,j) = B/C;
%             end 
%             
%             
%         end
%     end


    tmap = zeros(xf-xi +1,1);
    for i=xi:xf
        B = I(i) - Binf;
        C = J(i) - Binf;
        tmap(i-xi+1) = B/C;
%             tmap(i-xi+1,j-yi+1)  = -log(tmap(i-xi+1,j-yi+1));
%             tmap(i-xi+1,j-yi+1)  = real(tmap(i-xi+1,j-yi+1) );
%             tmap(i-xi+1,j-yi+1)  =  tmap(i-xi+1,j-yi+1)*8.2940496401;
%             tmap(i-xi+1,j-yi+1)  =  tmap(i-xi+1,j-yi+1)*dmap(i,j);
    end
    
    
    %figure;
    %imwrite(tmap
    %imshowdepth(tmap);
    
      

    %  tmap = (B./C);
    %tmap = B./C;
    % Multiply by the distance.

    %tmap = 1- tmap;
    %max(max(tmap))
    %tmap = tmap./max(max(tmap));
    %c = tmap(posx,posy);
    %figure
    %imshow(tmap);
% 
    dmap = dmap(xi:xf);
     tmap = log(tmap);
      tmap = real(tmap);
    % Considering the resolution of 4 pixel by millimiters and the distance
    % being considered in 
      
     tmap= tmap*8.2940496401;
   
     tmap = tmap./dmap;
%     %figure
%     %imshow(tmap);
%     %dmax = max(max(tmap));
%     z = tmap(:);
%     x =  [1:length(z)]';
%     y =  [1:length(z)]';
% 
%   %  sf = fit( [ y x], z, 'poly11');
% 
% 
%     %daltMap = tmap./dmax;
%     %c = 
%     %cmap = tmap./distance;
%     
%     %imshow(daltMap);
%     
% 
%     
%     %figure
%     %imshow(cmap);
%    
%     
%     %t = sum(sum(cmap))/(size(cmap,1)*size(cmap,2));
% 
%     
%     % vectorize tmap
 %    turb=0;
  %   for i=1:length(r)
%         turb
  %      turb = turb + tmap(r(i),c(i));
% 
  %   end
    c = sum(tmap)/size(tmap,1);
    %c = sum(sum(tmap))/(size(tmap,1)*size(tmap,2));
    % c = turb/length(r);
end
