function patch =samplePatch(image,px,py)
    posx = randsample(1:(size(image,1)-px),1,true);
    posy = randsample(1:(size(image,2)-py),1,true);
    
    patch= image(posx:(posx+px-1),posy:(posy+py-1),:);

end