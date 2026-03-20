function seq = WSIS(rep)
M = [rep.Cost]';
[n,~] = size(M);

% 将各指标映射到 [0,1]，确保 f1, f2, f3 平等对话
    M_min = min(M, [], 1);
    M_max = max(M, [], 1);
    M_range = M_max - M_min;
    M_range(M_range == 0) = 1; % 防止除以 0
    X = (M - repmat(M_min, n, 1)) ./ repmat(M_range, n, 1);

[w,~] = dynamicWeright(X);
PIS = min(X);
NIS = max(X);
dominance = zeros(1,n);
for i = 1:n
	dp=sqrt(sum(w.*((X(i,:)-PIS).^2)));
    dn=sqrt(sum(w.*((X(i,:)-NIS).^2)));
    dominance(i) = dn/(dp+dn);
end
[~,seq] = sort(dominance,'descend');
end


%%
function [w, cr] = dynamicWeright(P)
    [n, m] = size(P);
    
    % --- 关键修改：按列归一化，让 f1, f2, f3 在各自维度平等分布 ---
    % 这样 f1 的几千和 f3 的 0.1 就都有了同样的“话语权”
    col_min = min(P, [], 1);
    col_max = max(P, [], 1);
    col_range = col_max - col_min;
    col_range(col_range == 0) = 1; % 防止除以 0
    P_norm = (P - repmat(col_min, n, 1)) ./ repmat(col_range, n, 1);
    
    % 为了计算熵，需要保证概率和为 1 (加 eps 防 log(0))
    p_dist = (P_norm + eps) ./ repmat(sum(P_norm + eps, 1), n, 1);
    
    en = zeros(1, m);
    for j = 1:m
        en(j) = -sum(p_dist(:, j) .* log(p_dist(:, j))) / log(n + eps);
    end
    
    % 相关性计算 (增加防御逻辑)
    if n > 1
        R = corrcoef(P);
        R(isnan(R)) = 0; 
        cc = m - sum(abs(R), 2);
    else
        cc = ones(m, 1);
    end
    
    cr = (1 - en) .* cc';
    w = cr / (sum(cr) + eps); 
end