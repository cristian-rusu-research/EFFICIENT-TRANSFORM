function [U, X, positions, values, tus, err] = g_dla(Data, k0, m)

tic;
[n, ~] = size(Data);
[U, ~, ~] = svd(Data, 'econ');
X = omp_forortho(U'*Data, k0);

positions = zeros(2, m);
values = zeros(4, m);

% number of iterations
K = 50;

Z = Data*X';
scores_nuclear = zeros(n);
for i = 1:n
    for j = i+1:n
        T = Z([i j], [i j]);
        c1 = norm(T, 'fro')^2/2; c1_2 = c1^2; c2_2 = det(T)^2;
        scores_nuclear(i, j) = sqrt(c1 + sqrt(c1_2 - c2_2)) + sqrt(c1 - sqrt(c1_2 - c2_2)) - trace(T);
    end
end

for kk = 1:m
    [~, index_nuc] = max(scores_nuclear(:));
    [i_nuc, j_nuc] = ind2sub([n n], index_nuc);

    [Uu, ~, Vv] = svd(Z([i_nuc j_nuc], [i_nuc j_nuc]));
    GG = Uu*Vv';

    positions(1, kk) = i_nuc;
    positions(2, kk) = j_nuc;
    values(:, kk) = vec(GG);
    
    Z = applyGTransformOnRightTransp(Z, i_nuc, j_nuc, values(:, kk));

    for i = [i_nuc j_nuc]
        for j = i+1:n
            T = Z([i j], [i j]);
            c1 = norm(T, 'fro')^2/2; c1_2 = c1^2; c2_2 = det(T)^2;
            scores_nuclear(i, j) = sqrt(c1 + sqrt(c1_2 - c2_2)) + sqrt(c1 - sqrt(c1_2 - c2_2)) - trace(T);
        end
    end

    for j = [i_nuc j_nuc]
        for i = 1:j-1
            T = Z([i j], [i j]);
            c1 = norm(T, 'fro')^2/2; c1_2 = c1^2; c2_2 = det(T)^2;
            scores_nuclear(i, j) = sqrt(c1 + sqrt(c1_2 - c2_2)) + sqrt(c1 - sqrt(c1_2 - c2_2)) - trace(T);
        end
    end
end

P = Data;
for h = m:-1:1
    P = applyGTransformOnLeftTransp(P, positions(1, h), positions(2, h), values(:, h));
end
X = omp_forortho(P, k0);

err = zeros(K, 1);
for k = 1:K
    Z = Data*X';
    for h = m:-1:1
        Z = applyGTransformOnLeftTransp(Z, positions(1, h), positions(2, h), values(:, h));
    end
    
    for kk = 1:m
        Z = applyGTransformOnLeft(Z, positions(1, kk), positions(2, kk), values(:, kk));
        
        scores_nuclear = zeros(n);
        for i = 1:n
            for j = i+1:n
                T = Z([i j], [i j]);
                c1 = norm(T, 'fro')^2/2; c1_2 = c1^2; c2_2 = det(T)^2;
                scores_nuclear(i, j) = sqrt(c1 + sqrt(c1_2 - c2_2)) + sqrt(c1 - sqrt(c1_2 - c2_2)) - trace(T);
            end
        end
        
        [~, index_nuc] = max(scores_nuclear(:));
        [i_nuc, j_nuc] = ind2sub([n n], index_nuc);
        
        [Uu, ~, Vv] = svd(Z([i_nuc j_nuc], [i_nuc j_nuc]));
        GG = Uu*Vv';
        
        positions(1, kk) = i_nuc;
        positions(2, kk) = j_nuc;
        values(:, kk) = vec(GG);
        
        Z = applyGTransformOnRightTransp(Z, positions(1, kk), positions(2, kk), values(:, kk));
    end
    
    P = Data;
    for h = m:-1:1
        P = applyGTransformOnLeftTransp(P, positions(1, h), positions(2, h), values(:, h));
    end
    X = omp_forortho(P, k0);

    UX = X;
    for h = 1:m
        UX = applyGTransformOnLeft(UX, positions(1, h), positions(2, h), values(:, h));
    end
    err(k) = norm(Data-UX, 'fro')^2/norm(Data, 'fro')^2*100;
end
tus = toc;
