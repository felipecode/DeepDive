function T = calculateTurbidImage(J,c,dmap,averageCossine,Binf)


% c must be a matrix as big as the image

c_mat = c*ones(size(J));
dmap = 0.58*ones(size(J));

Binf_mat = Binf*ones(size(J));

Ed = J.*exp(-dmap.*c_mat);
Ef = conv2(Ed,psf(10,averageCossine,c),'same');
Eb = (Binf_mat - Binf_mat.*exp(-dmap.*c_mat));

%'Binf'
%sum(sum(Binf))/(size(Binf,1)*size(Binf,2))


T = Ed + Ef + Eb;
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