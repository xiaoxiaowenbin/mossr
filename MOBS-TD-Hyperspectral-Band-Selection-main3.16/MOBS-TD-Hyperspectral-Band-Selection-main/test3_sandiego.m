%% MOSSR Algorithm - PSO Engine Version 
% 核心思想：
% 1. 优化变量：权重矩阵 W (拉直为向量)，维度为 L*L
% 2. 目标函数：MOSSR 原版 (全局重构 + 目标重构 + TBS)
% 3. 优化算法：PSO (保留你提供的 Hydice 版本的所有逻辑细节)
tic;
clear; clc; close all;

%% 1. 核心配置
nRuns = 10;                  % 10轮循环
seed_base = 123;             % 种子 123-132
detector_Name = 'CEM';

% PSO 参数 (完全保留原版配置)
MaxT = 100;                  % 迭代次数
nPop = 100;                   % 种群大小
nRep = 20;                   % 档案大小
w = 0.5;                     % 惯性权重
c1 = 1.0;                    % 学习因子1
c2 = 1.0;                    % 学习因子2
nGrid = 4;                   % 网格数
maxrate = 0.2;               % 速度限制
mu = 0.1;                    % 变异概率

% ================= 文件名配置 =================
% 加载数据
load Sandiego.mat; % 假设变量名为 Sandiego
load PlaneGT.mat;  % 假设变量名为 PlaneGT

% --- 关键步骤 1: 空间裁剪 ---
[H_full, W_full, L_full] = size(Sandiego);
[H_gt, W_gt] = size(PlaneGT);
if H_full ~= H_gt || W_full ~= W_gt
    fprintf('正在裁剪图像左上角 %dx%d 区域以匹配 GT...\n', H_gt, W_gt);
    img_src = Sandiego(1:H_gt, 1:W_gt, :);
else
    img_src = Sandiego;
end
img_gt = PlaneGT;
[W, H, L] = size(img_src);

% --- 关键步骤 2: 剔除坏波段 (224 -> 189) ---
if L == 224
    disp('正在剔除坏波段 (保留 189 个有效波段)...');
    bad_bands = [1:6, 33:35, 97, 107:113, 153:166, 221:224];
    good_bands_idx = setdiff(1:224, bad_bands);
    img_src = img_src(:, :, good_bands_idx);
    [W, H, L] = size(img_src); % 更新波段数 L = 189
end

% --- 2.3 尺寸与归一化 ---
img_2d = reshape(img_src, W * H, L);

% 归一化 (MOSSR 必须步骤)
img_2d = double(img_2d);
img_min = min(img_2d);
img_max = max(img_2d);
img_norm = (img_2d - img_min) ./ (img_max - img_min + eps);
img = img_norm; 

% 提取目标
d = get_target(img, img_gt);
if size(d, 1) > size(d, 2), d = d'; end 

% 预计算相关矩阵 (MOSSR 核心)
disp('预计算相关矩阵 Y''Y...');
YtY = img' * img;

% VD 估计确定 K
t_vd = 1e-4; 
fprintf('执行 VD 估计...\n');
% VD估计需要使用原始数据的统计特性，这里传入 double(img_src)
hfc_vd = HFC1(double(img_src), t_vd); 
nwhfc_vd = NWHFC(double(img_src), t_vd);
K = 2 * max(hfc_vd, nwhfc_vd); 
fprintf('最终选择的波段数量 (K): %d\n', K);

%% 3. 初始化结果存储
AUC_PFPD = zeros(nRuns, 1);
AUC_tauPD = zeros(nRuns, 1);
AUC_tauPF = zeros(nRuns, 1);
selected_bands_all = cell(nRuns, 1);

%% 4. 主循环 (PSO-MOSSR)
for runIdx = 1:nRuns
    fprintf('\n==================== 第 %d/%d 轮运行 (PSO-MOSSR) ====================\n', runIdx, nRuns);
    
    current_seed = seed_base + runIdx - 1;
    rng(current_seed, 'twister');
    fprintf('本轮随机种子: %d\n', current_seed);
    
    %% 4.1 变量定义 (MOSSR 的变量是 W 矩阵)
    num_vars = L * L; 
    VarSize = [1 num_vars];
    VarMin = 0; 
    VarMax = 1;
    
    % 定义 MOSSR 的适应度函数
    CostFunction = @(x) mossr_fitness_test2(x, YtY, L, d);

    %% 4.2 PSO 初始化
    empty_particle.Position = [];
    empty_particle.Velocity = [];
    empty_particle.Cost = [];
    empty_particle.Best.Position = [];
    empty_particle.Best.Cost = [];
    empty_particle.IsDominated = [];
    empty_particle.GridIndex = [];
    empty_particle.GridSubIndex = [];
    
    pop = repmat(empty_particle, nPop, 1);
    
    % [针对 WHU 数据的噪声屏蔽]
    % WHU 数据通常首尾波段较差，这里屏蔽前3个和后3个
    noise_bands = [1:3, (L-2):L];
    mask_matrix = ones(L, L);
    mask_matrix(noise_bands, :) = 0; 
    mask_matrix(:, noise_bands) = 0;
    mask_flat = reshape(mask_matrix, 1, num_vars);
    valid_indices = find(mask_flat == 1);
    
    disp('初始化 PSO 粒子 (W矩阵)...');
    for i = 1:nPop
        % 稀疏初始化
        pop(i).Position = zeros(VarSize);
        num_active = floor(length(valid_indices) * 0.2);
        active_idx = valid_indices(randperm(length(valid_indices), num_active));
        pop(i).Position(active_idx) = rand(1, num_active);
        
        pop(i).Velocity = zeros(VarSize);
        
        pop(i).Cost = CostFunction(pop(i).Position);
        
        pop(i).Best.Position = pop(i).Position;
        pop(i).Best.Cost = pop(i).Cost;
    end
    
    % 建立档案
    pop = DetermineDomination(pop);
    rep = pop(~[pop.IsDominated]);
    rep = GridIndex(rep, nGrid);
    
    %% 4.3 PSO 主迭代
    disp('开始 PSO 优化权重矩阵 W...');
    for it = 1:MaxT
        pm = (1-(it-1)/(MaxT-1))^(1/mu); 
        
        for i = 1:nPop
            leader = SelectLeader(rep);
            pop(i).Velocity = w * pop(i).Velocity ...
                + c1 * rand(VarSize) .* (pop(i).Best.Position - pop(i).Position) ...
                + c2 * rand(VarSize) .* (leader.Position - pop(i).Position);
            
            % [保留原版逻辑] 速度限制与 L 相关
            pop(i).Velocity = max(pop(i).Velocity, -maxrate*L);
            pop(i).Velocity = min(pop(i).Velocity, maxrate*L);
            
            pop(i).Position = pop(i).Position + pop(i).Velocity;
            
            % 边界处理与物理屏蔽
            pop(i).Position = max(pop(i).Position, VarMin);
            pop(i).Position = min(pop(i).Position, VarMax);
            pop(i).Position(~mask_flat) = 0; % 关键：确保噪声波段不参与
            
            % 更新基础适应度
            pop(i).Cost = CostFunction(pop(i).Position);
            
            % [保留原版逻辑] 变异机制
            if rand < pm
                nMutate = max(1, round(0.01 * length(valid_indices))); 
                mutate_sub_idx = valid_indices(randperm(length(valid_indices), nMutate));
                
                pop(i).Position(mutate_sub_idx) = rand(1, nMutate);
                
                pop(i).Cost = CostFunction(pop(i).Position);
            end
            
            % 更新个体最优 PBest
            if Dominates(pop(i).Cost, pop(i).Best.Cost)
                pop(i).Best.Position = pop(i).Position;
                pop(i).Best.Cost = pop(i).Cost;
            elseif ~Dominates(pop(i).Best.Cost, pop(i).Cost)
                if rand < 0.5
                    pop(i).Best.Position = pop(i).Position;
                    pop(i).Best.Cost = pop(i).Cost;
                end
            end
        end
        
        % 维护档案
        rep = [rep; pop]; 
        rep = DetermineDomination(rep);
        rep = rep(~[rep.IsDominated]);
        rep = GridIndex(rep, nGrid);
        
        % 删除多余档案成员
        if numel(rep) > nRep
            try
                seq = WSIS(rep);
                Extra = numel(rep) - nRep;
                for e = 1:Extra
                    rep = DeleteRepMemebr(rep, seq);
                end
            catch
                Extra = numel(rep) - nRep;
                rep(randperm(numel(rep), Extra)) = [];
            end
        end
        
        % 衰减惯性
        w = w * 0.99;
    end
    
    %% 4.4 波段选择
    disp('优化完成，执行 MOSSR 波段选择...');
    
    x_rep = vertcat(rep.Position);
    f_rep = vertcat(rep.Cost);
    
    final_selected_bands = select_bands_test2(x_rep, f_rep, L, K);
    selected_bands_all{runIdx} = final_selected_bands;
    
    %% 4.5 CEM 验证
    img_sel = img(:, final_selected_bands);
    d_sel = d(final_selected_bands);
    
    detectmap = reshape(detector(img_sel, d_sel', detector_Name), W, H);
    
    % 最后一轮画图
    if runIdx == nRuns
        figure(1); imagesc(detectmap); 
        title(['第', num2str(runIdx), '轮检测图 (PSO-MOSSR - WHU)']); 
        colormap jet; colorbar;
    end
    
    % AUC
    det_map_vec = detectmap(:);
    GT_vec = img_gt(:);
    [AUC, ~] = cal_AUC(det_map_vec, GT_vec, 1, 1);
    
    AUC_PFPD(runIdx) = AUC.PFPD;
    AUC_tauPD(runIdx) = AUC.tauPD;
    AUC_tauPF(runIdx) = AUC.tauPF;
    
    fprintf('  AUC (PF-PD): %.6f\n', AUC.PFPD);
end

%% 5. 结果汇总
fprintf('\n==================== 最终统计 ====================\n');
fprintf('平均 AUC (PF-PD): %.6f (Std: %.6f)\n', mean(AUC_PFPD), std(AUC_PFPD));
fprintf('最大 AUC (PF-PD): %.6f\n', max(AUC_PFPD));
