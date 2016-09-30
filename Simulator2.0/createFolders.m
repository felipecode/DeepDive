function createFolders(root,databaseName)
mkdir(sprintf('%s/%s/Validation',root,databaseName)); 

mkdir(sprintf('%s/%s/ValidationTransmission',root,databaseName));

mkdir(sprintf('%s/%s/ValidationGroundTruth',root,databaseName)); 

mkdir(sprintf('%s/%s/Test',root,databaseName));

mkdir(sprintf('%s/%s/TestGroundTruth',root,databaseName)); 

mkdir(sprintf('%s/%s/TestTransmission',root,databaseName));

mkdir(sprintf('%s/%s/Training',root,databaseName));

mkdir(sprintf('%s/%s/GroundTruth',root,databaseName)); 

mkdir(sprintf('%s/%s/Transmission',root, databaseName));
