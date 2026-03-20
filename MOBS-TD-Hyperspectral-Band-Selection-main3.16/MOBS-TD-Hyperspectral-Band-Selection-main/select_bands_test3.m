function selected_bands = select_bands_test3(x_pareto, f_pareto, num_bands, K, snr_vec, target_d)
    % 核心思路：
    % 1. 档案集成：不只选一个最优解，而是对整个非支配档案进行加权投票。
    % 2. 自适应排斥：避免选取的波段过于集中在某一个谱段。
    % 3. 稳健性增强：利用多个解的统计特性（均值）作为打分基础。

    target_d = target_d(:);
    [N, L2] = size(x_pareto);
    
    % ==================== 1. 档案解集成 (Ensemble) ====================
    % 将所有非支配解的权重矩阵取平均，平滑掉单个解的波动
    all_W = zeros(num_bands, num_bands);
    for i = 1:N
        all_W = all_W + reshape(x_pareto(i, :), num_bands, num_bands);
    end
    W_mean = all_W / N; 
    
    % 计算加权后的能量（反映了多目标下的共同关注点）
    w_energy = sum(W_mean.^2, 2);
    w_energy_norm = (w_energy - min(w_energy)) / (max(w_energy) - min(w_energy) + eps);
    
    % ==================== 2. 鲁棒的先验融合 ====================
    % 引入指数加权，使目标特征更突出，避免微弱特征干扰
    target_prior = abs(target_d);
    target_prior_norm = (target_prior - min(target_prior)) / (max(target_prior) - min(target_prior) + eps);
    
    % 使用非线性融合：乘积通常会导致小值被放大，改为加权和或改进的幂乘
    % 这里使用 0.7*Energy + 0.3*Prior，平衡重构性能与目标相关性
    band_scores = 0.7 * w_energy_norm + 0.3 * (target_prior_norm.^0.5);
    
    % ==================== 3. 噪声与边界处理 ====================
    % 动态计算阈值：使用 25% 分位数代替 median，更稳健
    snr_threshold = quantile(snr_vec, 0.25);
    noise_bands = find(snr_vec < snr_threshold);
    edge_bands = [1:5, (num_bands-4):num_bands]; % 适当扩大边缘静默区
    invalid_bands = unique([noise_bands; edge_bands']);
    band_scores(invalid_bands) = -inf;
    
    % ==================== 4. 带惩罚机制的贪婪选择 ====================
    selected_bands = [];
    current_scores = band_scores;
    
    % 初始排斥半径 (动态缩小)
    radius = floor(num_bands / K); 
    
    for k = 1:K
        [~, idx] = max(current_scores);
        if current_scores(idx) == -inf, break; end
        
        selected_bands = [selected_bands; idx];
        
        % 自适应惩罚：不仅设为-inf，还对邻域进行梯度削减，防止波段过于靠拢
        for d = 1:radius
            penal = (1 - d/radius)^2; % 二次惩罚函数
            if idx-d > 0, current_scores(idx-d) = current_scores(idx-d) * penal; end
            if idx+d <= num_bands, current_scores(idx+d) = current_scores(idx+d) * penal; end
        end
        current_scores(idx) = -inf;
    end
    
    % ==================== 5. 排序与最终校验 ====================
    selected_bands = sort(selected_bands);
    
    % 补充缺失波段：如果物理意义上的波段太少，补充SNR最高的波段
    if length(selected_bands) < K
        [~, sorted_snr_idx] = sort(snr_vec, 'descend');
        for i = 1:length(sorted_snr_idx)
            if ~ismember(sorted_snr_idx(i), selected_bands) && ~ismember(sorted_snr_idx(i), invalid_bands)
                selected_bands = [selected_bands; sorted_snr_idx(i)];
            end
            if length(selected_bands) == K, break; end
        end
    end
    selected_bands = sort(selected_bands);
end