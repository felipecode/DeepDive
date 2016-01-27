function T = calculateTurbidImageFixD(J,c,distance,averageCossine,Binf)


% c must be a matrix as big as the image

c_mat = c*ones(size(J));

c_mat = c_mat + 0.1*randn(size(c_mat));


dmap = distance*ones(size(J));

Binf_mat = Binf*ones(size(J));

Ed = J.*exp(-dmap.*c_mat);

%Ed = directComponent(J, c_mat,dmap);
%Ed = Ed./max(max(Ed));


Ef = conv2(Ed,psf(10,averageCossine,c),'same');
Eb = (Binf_mat - Binf_mat.*exp(-dmap.*c_mat));

%'Binf'
%sum(sum(Binf))/(size(Binf,1)*size(Binf,2))


T = Ed + Ef  + Eb;
%T = T./max(max(T));
%T = Eb;

%figure;
%imshow(T);

end

%c_mat = c*ones(size(J));

%Ed = J.*exp(-distance*c_mat);
%Ef = conv2(Ed,psf(10,averageCossine,c),'same');
%Eb = Binf - Binf.*exp(-distance*c_mat);

%T = Ed + Ef + Eb;