%% 附录函数：HFC-VD 算法实现
function k = calc_vd_hfc(data, pf)
% CALC_VD_HFC 计算虚拟维度 (Virtual Dimensionality)
% 输入:
%   data - [Pixels x Bands] 的高光谱数据矩阵
%   pf   - 虚警概率 (False Alarm Rate)，例如 10^-3
% 输出:
%   k    - 估计的信号源数量 (虚拟维度)

    [N, L] = size(data); % N: 像素数, L: 波段数
    
    % 1. 计算相关矩阵 (Correlation Matrix) R
    % R = (1/N) * (Y' * Y)
    R = (data' * data) / N;
    
    % 2. 计算协方差矩阵 (Covariance Matrix) K_cov
    % 首先去中心化
    mu = mean(data);
    data_centered = data - mu;
    K_cov = (data_centered' * data_centered) / N;
    
    % 3. 特征值分解
    [~, D_R] = eig(R);
    [~, D_K] = eig(K_cov);
    
    % 提取特征值并降序排列
    eig_R = sort(diag(D_R), 'descend');
    eig_K = sort(diag(D_K), 'descend');
    
    % 4. 计算 HFC 统计量
    % 假设噪声为高斯白噪声，特征值差异的方差估计
    var_diff = (2/N) * (eig_R.^2 + eig_K.^2);
    sigma_diff = sqrt(var_diff);
    
    % 计算阈值 Tau
    % 使用标准正态分布的逆函数计算临界值
    % z_score 对应于单尾检验
    z_score = norminv(1 - pf); 
    tau = z_score * sigma_diff;
    
    % 5. 确定 K 值
    % 如果 (eig_R - eig_K) > tau，则认为该维度包含信号
    diff_eig = eig_R - eig_K;
    
    % 统计满足条件的数量
    k = sum(diff_eig > tau);
    
    % 修正：防止 K 为 0 (至少保留 1 个特征)
    if k == 0
        k = 1;
    end
end