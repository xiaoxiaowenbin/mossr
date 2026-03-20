function f = mossr_fitness_test2(x, YtY, num_bands, target_d)
    % 1. 恢复 W 矩阵
    W = reshape(x, num_bands, num_bands);
    W(1:num_bands+1:end) = 0; % 对角线置零
    
    % 2. 目标 1: 全局重构误差 ||Y - YW||_F^2
    I = eye(num_bands);
    diff_term = I - W;
    temp = YtY * diff_term;
    f1 = sum(dot(diff_term, temp, 1)); 
    
    d_recon = target_d * W;
    diff_d = target_d - d_recon;
    f2 = sum(diff_d.^2);
    
    % 4. 目标 3: TBS (Target-Background Separation)
    R_hat = W' * YtY * W;
    
    % [微调 A]: 更稳健的对角加载 (用 trace 代替 mean diag，加上 1e-6 底线)
    lambda = 1e-4 * (trace(R_hat) / num_bands) + 1e-6; 
    R_reg = R_hat + lambda * eye(num_bands);
    
    inv_quad_form = (target_d / R_reg) * target_d';
    
    % [微调 B]: 加上 abs 防止数值误差导致负数，引发优化器崩溃
    f3 = 1 / (abs(inv_quad_form) + eps); 

    % [微调 C]: 权重平衡！给 f3 乘上 10^5，让它能和 f1, f2 坐在同一张桌子上谈判
    f = [f1, f2, f3 ]; 
end