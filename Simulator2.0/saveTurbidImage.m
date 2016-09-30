function saveTurbidImage(turbidImage,groundTruth,transmission,basePath,databaseName,baseImageIndex,TurbidImageIndex)
    databaseFullPath=strcat(basePath,databaseName);
    if mod(baseImageIndex,100) == 9
        imwrite(turbidImage,sprintf('%s/Validation/%d.jpg',databaseFullPath,TurbidImageIndex,'Compression','none'));
        imwrite(groundTruth,sprintf('%s/ValidationGroundTruth/%d.jpg',databaseFullPath,TurbidImageIndex,'Compression','none'));
        imwrite(transmission,sprintf('%s/ValidationTransmission/%d.jpg',databaseFullPath,TurbidImageIndex,'Compression','none'));

    elseif mod(baseImageIndex,100) == 8
        imwrite(turbidImage,sprintf('%s/Test/%d.jpg',databaseFullPath,TurbidImageIndex,'Compression','none'));
        imwrite(groundTruth,sprintf('%s/TestGroundTruth/%d.jpg',databaseFullPath,TurbidImageIndex,'Compression','none'));   
        imwrite(transmission,sprintf('%s/TestTransmission/%d.jpg',databaseFullPath,TurbidImageIndex,'Compression','none'));
    else
        imwrite(turbidImage,sprintf('%s/Training/%d.jpg',databaseFullPath,TurbidImageIndex,'Compression','none'));
        imwrite(groundTruth,sprintf('%s/GroundTruth/%d.jpg',databaseFullPath,TurbidImageIndex,'Compression','none'));    
        imwrite(transmission,sprintf('%s/Transmission/%d.jpg',databaseFullPath,TurbidImageIndex,'Compression','none'));
    end
end