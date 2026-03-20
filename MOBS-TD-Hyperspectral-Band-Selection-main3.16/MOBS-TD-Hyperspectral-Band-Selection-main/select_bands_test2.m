function selected_bands = select_bands_test2(x_pareto, f_pareto, num_bands, K)
    num_solutions = size(x_pareto, 1);
    f3_vals = f_pareto(:, 3);
    [~, sort_idx] = sort(f3_vals, 'ascend');
    
    % 取前 40% 的解
    num_elite = max(1, round(num_solutions * 0.4));
    elite_indices = sort_idx(1:num_elite);
    x_elite = x_pareto(elite_indices, :);
    
    % 累积投票
    band_scores = zeros(num_bands, 1);
    for i = 1:num_elite
        W = reshape(x_elite(i, :), num_bands, num_bands);
        band_scores = band_scores + sum(W.^2, 2);
    end
    
    % ==========================================================
    % [微调 D]: 噪声波段静默 (针对高光谱常见特性)
    % 强制屏蔽前 5 个和最后 5 个波段，不让它们参与竞选
    noise_bands = [1:5, (num_bands-4):num_bands];
    band_scores(noise_bands) = -inf; 
    % ==========================================================
    
    selected_bands = [];
    temp_scores = band_scores;
    exclusion_window = 2; % 保持你原来的 2，这个值对于连续特征很好
    
    for k = 1:K
        [max_score, best_idx] = max(temp_scores);
        
        if max_score == -inf
            break; 
        end
        
        selected_bands = [selected_bands; best_idx];
        
        % 抑制邻域 (保持你的原版逻辑)
        low_bound = max(1, best_idx - exclusion_window);
        up_bound = min(num_bands, best_idx + exclusion_window);
        temp_scores(low_bound:up_bound) = -inf;
    end
    
    selected_bands = sort(selected_bands);
    
    % 随机填补
    if length(selected_bands) < K
        % 填补时也要避开噪声波段
        valid_pool = setdiff(1:num_bands, [selected_bands', noise_bands]);
        n_need = K - length(selected_bands);
        if ~isempty(valid_pool)
            selected_bands = sort([selected_bands; valid_pool(1:min(n_need, length(valid_pool)))']);
        end
    end
end