dimensaoMenor = [16,16];
dimensaoMaior = [224,224];

depthMenor = depths(1:dimensaoMenor(1),1:dimensaoMaior(2),:);
for i=1:dimensaoMenor(1):dimensaoMaior(1)-dimensaoMenor(1)
    depthMenor =cat(3,depthMenor,depths(i:i+dimensaoMenor(1)-1,1:dimensaoMenor(2),:));
end
for i=1:dimensaoMenor(1):dimensaoMaior(1)-dimensaoMenor(1)
    for j=dimensaoMenor(2)+1:dimensaoMenor(2):dimensaoMaior(2)-dimensaoMenor(2)
    depthMenor =cat(3,depthMenor,depths(i:i+dimensaoMenor(1)-1,j:j+dimensaoMenor(2)-1,:));
    end
end

imagesMenor = images(1:dimensaoMenor(1),1:dimensaoMenor(2),:,:);
for i=1:dimensaoMenor(1):dimensaoMaior(1)-dimensaoMenor(1)
    imagesMenor =cat(4,imagesMenor,images(i:i+dimensaoMenor(1)-1,1:dimensaoMenor(2),:,:));
end
for i=1:dimensaoMenor(1):dimensaoMaior(1)-dimensaoMenor(1)
    for j=dimensaoMenor(2)+1:dimensaoMenor(2):dimensaoMaior(2)-dimensaoMenor(2)
    imagesMenor =cat(4,imagesMenor,images(i:i+dimensaoMenor(1)-1,j:j+dimensaoMenor(2)-1,:,:));
    end
end