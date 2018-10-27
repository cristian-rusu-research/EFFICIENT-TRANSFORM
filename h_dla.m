function [U, X, theUs, tus, err] = h_dla(Data, k0, m)

tic;
[n, ~] = size(Data);

% the initialization of Householder reflectors
theUs = zeros(n, m);

% the initialization of X
[U, ~, ~] = svd(Data, 'econ');
X = omp_forortho(U'*Data, k0);

% number of iterations
K = 30;
err = zeros(K, 1);
for k = 1:K
    
    R = X*Data';
    for j = m:-1:2
        R = applyReflectorOnRight(R, theUs(:,j));
    end
    
    for h = 1:m
        warning('off','all');
        [Vv, Dd] = eigs(R+R', 1, 'SA');
        warning('on','all');
        
        if (Dd < 0)
            theUs(:, h) = Vv;
        else
            theUs(:, h) = 0;
        end
        
        if (h < m)
            R = applyReflectorOnLeft(theUs(:,h), R);
            R = applyReflectorOnRight(R, theUs(:,h+1));
        end
    end
    
    P = Data;
    for j = m:-1:1
        P = applyReflectorOnLeft(theUs(:,j), P);
    end
    X = omp_forortho(P, k0);
    
    UX = X;
    for j = 1:m
        UX = applyReflectorOnLeft(theUs(:,j), UX);
    end
    
    err(k) = norm(Data-UX, 'fro')^2/norm(Data, 'fro')^2*100;
end
tus = toc;
