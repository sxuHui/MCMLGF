function [LHs] = Hs2LHs_svd(Hs, eta, knn_size,m)
nView = length(Hs);

% Ss = cell(1, nView);
% Ls = cell(1, nView);
% HLs = cell(1, nView);
LHs = cell(nView, nView);
for iView = 1:nView
    [~, Xa] = litekmeans(Hs{iView}, m, 'Replicates', 1);   
    B = ConstructBP_pkn(Hs{iView}, Xa, 'nNeighbor', knn_size);
    idx = sum(B, 1) > 0;
    B = B(:, idx);
    P = B * diag(sum(B, 1).^(-.5));
    [U, Lambda, V] = svds(P, min(size(P))); % n * m *m
    Lm = eye(size(Lambda, 1)) - Lambda * Lambda';
    HL2 = expm(-eta * Lm); % m^3
    HL2 = (HL2 + HL2')/2;
    for iView2 = 1:nView
        tmp1 = U' * Hs{iView2}; % m * n * d
        tmp2 = HL2 * tmp1; % m * m * d
        LHs{iView2, iView} = U * tmp2; % n m d
    end
end

end