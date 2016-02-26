function forward = sampleDistance(decay,maxForward)

    numvec = 80*exp(decay*(0:-0.1:-maxForward*0.1));
    randomDistVec =[];
    for i=1:maxForward
        randomDistVec = [randomDistVec;i*ones(uint32(round(numvec(i))),1)/10];

    end

    forward = randsample(randomDistVec,1);
end