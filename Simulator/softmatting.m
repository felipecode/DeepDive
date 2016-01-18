%% Matting the transmission map
% Code in this section was partly inspired by code originally from
% http://www.soe.ucsc.edu/classes/ee264/Winter10/Proj6-Code.zip


function t = softmatting(trans_est,im)

    win_size = 3; % window size
    win_els = win_size.^2; % number of window elements
    l_els = win_els.^2; % number of elements calculated in each iteration

    win_bord = floor(win_size./2); 

    e = 0.000001; 

    [m,n,c] = size(im);
    numpix = m*n;

    k = reshape(1:numpix, m, n);
    U = eye(win_size);
    D = eye(win_els);

    num_els = l_els*(m-2*win_bord)*(n-2*win_bord); 

    ind_i  = ones(1,num_els);
    ind_j = ind_i;

    els = zeros(1,num_els);

    count = 0;

    'Aqui4'
    for x = (1 + win_bord):(n - win_bord)
        for y = (1 + win_bord):(m - win_bord)
		
            wk = reshape(im(y-win_bord:y+win_bord,x-win_bord:x+win_bord,:), win_els, c);
                
            w_ind = reshape(k(y-win_bord:y+win_bord,x-win_bord:x+win_bord), 1, win_els);
                
            [i j] = meshgrid(w_ind, w_ind);
            
            i = reshape(i,1,l_els);
            j = reshape(j,1,l_els);
            
            ind_i((count*l_els + 1):(count*l_els+l_els)) = i;
            ind_j((count*l_els + 1):(count*l_els+l_els)) = j;

            win_mu = mean(wk)';

            win_cov = wk'*wk/win_els-win_mu*win_mu';

            dif = wk' - repmat(win_mu,1,win_els);
            
            elements = D - (1 + dif(:,1:win_els)'*inv(...
                win_cov + e./win_els.*U)*dif(:,1:win_els))...
                ./win_els;

            els((count*l_els + 1):(count*l_els+l_els)) = ...
                reshape(elements,1,l_els);
            
            count = count + 1;
        end
    end


    L = sparse(ind_i, ind_j, els, numpix, numpix);

    %% generate refined transmission map
    'Aqui6'
    % recommended value from HST paper
    lambda = .0001;
    % equation 15 from HST

    a=trans_est(:) .* lambda;
    'Aqui6_1'
    b=lambda .* speye(size(L));
    'Aqui6_2'
    soma=L + b;
    'Aqui6_3'
    t = (soma) \ a;

    'Aqui61'
    t = t - min(t(:));

    'Aqui62'
    t = t ./ (max(t(:)));
    'Aqui63'
    t = t .* (max(trans_est(:)) - min(trans_est(:))) + min(trans_est(:));
    'Aqui64'
    t = reshape(t, size(trans_est));

end
