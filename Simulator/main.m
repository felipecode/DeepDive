  
image = imvec{18};
I1 = imvec{1};
distance = 0.58;
%dmapR = calculateDmap(image(:,:,1),distance); 


[I, spImage] = spAverageImageGray(image,164);
dmapR = calculateDmap(image(:,:,1),distance); 
  
[dmapInput] = spAverageImageWSpImageGray(dmapR,spImage);
dmapOutput = dmapInput;


input = imvec{1};
[J] = spAverageImageWSpImageGray(I1, spImage);
 
BinfR=0.2;
BinfG=1;
BinfB=0.7;
%[J, spImage] = spAverageImage(imvec{i} ,96);
T =simulateTurbidImage(input,J,I,BinfR,BinfG,BinfB,dmapInput,dmapOutput);
