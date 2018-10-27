function [U, X, D, positions, values, tus, err] = r_dla(Data, k0, m, X)

tic;
[n, ~] = size(Data);

% first set of iterations
K1 = 20;
% second set of iterations
K2 = 3;

if (isempty(X))
    [U, ~, ~] = svd(Data, 'econ');
    X = omp_forortho(U'*Data, k0);
end
D = eye(n);

positions = zeros(2, m);
values = zeros(4, m);

perf = +inf;
bestU = [];
bestX = [];
bestpositions = [];
bestvalues = [];

err = zeros(K1, 1);
for ii = 1:K1
    Z = Data*X';
    W = X*X';
    scores_nuclear = zeros(n);
    for i = 1:n
        for j = i+1:n
            scores_nuclear(i,j) = W(i,i) + W(j,j) + 1/(W(i,i)*W(j,j)-W(i,j)*W(j,i))*( W(i,i)*(Z(i, j)^2 + Z(j, j)^2 ) + W(j,j)*(Z(i,i)^2 + Z(j, i)^2) - (Z(i,i)*Z(i,j)+Z(j,i)*Z(j,j))*(W(i,j)+W(j,i)) ) -  2*(Z(i, i) + Z(j, j));
        end
    end

    U = eye(n);
    workingX = X;
    for kk = 1:m
        [~, index_nuc] = max(scores_nuclear(:));
        [i_nuc, j_nuc] = ind2sub([n n], index_nuc);

        XX = workingX([i_nuc j_nuc], :);
        YY = Data([i_nuc j_nuc], :);
        GG = (XX'\YY')';

        positions(1, kk) = i_nuc;
        positions(2, kk) = j_nuc;
        values(:, kk) = vec(GG);
        
        U = applyGTransformOnLeft(U, positions(1, kk), positions(2, kk), values(:, kk));
        workingX = applyGTransformOnLeft(workingX, positions(1, kk), positions(2, kk), values(:, kk));
        
        Z = applyGTransformOnRightTransp(Z, positions(1, kk), positions(2, kk), values(:, kk));
        W = applyGTransformOnLeft(W, positions(1, kk), positions(2, kk), values(:, kk));
        W = applyGTransformOnRightTransp(W, positions(1, kk), positions(2, kk), values(:, kk));
     
        for i = [i_nuc j_nuc]
            for j = i+1:n
                scores_nuclear(i,j) = W(i,i) + W(j,j) + 1/(W(i,i)*W(j,j)-W(i,j)*W(j,i))*( W(i,i)*(Z(i, j)^2 + Z(j, j)^2 ) + W(j,j)*(Z(i,i)^2 + Z(j, i)^2) - (Z(i,i)*Z(i,j)+Z(j,i)*Z(j,j))*(W(i,j)+W(j,i)) ) -  2*(Z(i, i) + Z(j, j));
            end
        end
        
        for j = [i_nuc j_nuc]
            for i = 1:j-1
                scores_nuclear(i,j) = W(i,i) + W(j,j) + 1/(W(i,i)*W(j,j)-W(i,j)*W(j,i))*( W(i,i)*(Z(i, j)^2 + Z(j, j)^2 ) + W(j,j)*(Z(i,i)^2 + Z(j, i)^2) - (Z(i,i)*Z(i,j)+Z(j,i)*Z(j,j))*(W(i,j)+W(j,i)) ) -  2*(Z(i, i) + Z(j, j));
            end
        end
    end
    
    for jj = 1:n
        D(jj, jj) = 1/norm(U(:,jj));
        U(:,jj) = U(:,jj)/norm(U(:,jj));
    end
    X = omp(U'*Data, U'*U, k0);
    
    err(ii) = norm(Data-U*X, 'fro')^2/norm(Data, 'fro')^2*100;
    
    if (err(ii) < perf)
        bestU = U;
        bestD = D;
        bestX = X;
        bestpositions = positions;
        bestvalues = values;
        perf = err(ii);
    end
end

U = bestU;
D = bestD;
X = bestX;
positions = bestpositions;
values = bestvalues;

err = zeros(K2, 1);
y = vec(Data);
for ii = 1:K2
    
    A = X;
    
    B = eye(n);
    for jj = m:-1:1
        B = applyGTransformOnRight(B, positions(1, jj), positions(2, jj), values(:, jj));
    end
        
    for kk = 1:m
        B = applyGTransformOnRightInverse(B, positions(1, kk), positions(2, kk), values(:, kk));
        
        y_now = y;
        for jj = setdiff(1:n, [positions(1, kk) positions(2, kk)]);
            y_now = y_now - kron(A(jj, :)', B(:, jj));
        end
        
        C = [kron(A(positions(1, kk), :)', B(:, positions(1, kk))) kron(A(positions(1, kk), :)', B(:, positions(2, kk))) kron(A(positions(2, kk), :)', B(:, positions(1, kk))) kron(A(positions(2, kk), :)', B(:, positions(2, kk)))];
        GG = C\y_now;
        values(:, kk) = vec(GG);
        
        A = applyGTransformOnLeft(A, positions(1, kk), positions(2, kk), values(:, kk));
    end

    U = eye(n);
    for kk = 1:m
        U = applyGTransformOnLeft(U, positions(1, kk), positions(2, kk), values(:, kk));
    end
    
    for jj = 1:n
        D(jj, jj) = 1/norm(U(:,jj));
        U(:,jj) = U(:,jj)/norm(U(:,jj));
    end
    X = omp(U'*Data, U'*U, k0);
    
    err(ii) = norm(Data-U*X, 'fro')^2/norm(Data, 'fro')^2*100;
    
    if (err(ii) < perf)
        bestU = U;
        bestD = D;
        bestX = X;
        bestpositions = positions;
        bestvalues = values;
        perf = err(ii);
    end
end

U = bestU;
D = bestD;
X = bestX;
positions = bestpositions;
values = bestvalues;

tus = toc;
