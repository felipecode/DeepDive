function Ed = calculateGTImageFixD(J,c,distance,minpixel)


% c must be a matrix as big as the image


c_mat = c*ones(size(J));

c_mat = c_mat + 0.01*randn(size(c_mat));


dmap = distance*ones(size(J));



Ed = J.*exp(-dmap.*c_mat);

%Ed = directComponent(J, c_mat,dmap);
Ed(Ed<minpixel*0.00392156862) = 0;
Ed = Ed./max(max(Ed));
%figure; 
%imshow(Ed);

end

%c_mat = c*ones(size(J));

%Ed = J.*exp(-distance*c_mat);
%Ef = conv2(Ed,psf(10,averageCossine,c),'same');
%Eb = Binf - Binf.*exp(-distance*c_mat);

%T = Ed + Ef + Eb;