%% MOSSR Algorithm - PSO Engine Version (保留 MOSSR 核心逻辑)
% 核心思想：
% 1. 优化变量：权重矩阵 W (拉直为向量)，维度为 L*L
% 2. 目标函数：MOSSR 原版 (全局重构 + 目标重构 + TBS)
% 3. 优化算法：替换为 PSO (使用你提供的 SelectLeader, DetermineDomination 等)
tic;
clear; clc; close all;

%% 1. 核心配置
nRuns = 10;                  % 10轮循环
seed_base = 123;             % 种子 123-132
detector_Name = 'CEM';

% PSO 参数 (针对连续变量 W 优化进行调整)
MaxT = 100;                  % 迭代次数
nPop = 100;                   % 种群大小 (W矩阵很大，建议不要太大)
nRep = 30;                   % 档案大小
w = 0.5;                     % 惯性权重
c1 = 1.0;                    % 学习因子1
c2 = 1.0;                    % 学习因子2
nGrid = 4;                   % 网格数 (用于 SelectLeader)
maxrate = 0.2;               % 速度限制
mu = 0.1;         % mutation  probability regulator

%% 2. 数据加载与 MOSSR 预处理 (这一步完全保留 MOSSR 原样)
disp('正在加载数据...');
load abu-airport-2.mat; 
img_src = data;
img_gt = map;

[W, H, L] = size(img_src);
img_2d = reshape(img_src, W * H, L);

% 归一化 (MOSSR 必须步骤)
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
hfc_vd = HFC1(img_src, t_vd);
nwhfc_vd = NWHFC(img_src, t_vd);
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
    % 注意：这里不再是优化索引，而是优化权重！
    num_vars = L * L; 
    VarSize = [1 num_vars];
    VarMin = 0; 
    VarMax = 1;
    
    % 定义 MOSSR 的适应度函数 (调用下方的 mossr_fitness)
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
    
    % 噪声波段掩膜 (MOSSR 物理屏蔽)
    noise_bands = [1:8, (L-5):L];
    mask_matrix = ones(L, L);
    mask_matrix(noise_bands, :) = 0; 
    mask_matrix(:, noise_bands) = 0;
    mask_flat = reshape(mask_matrix, 1, num_vars);
    valid_indices = find(mask_flat == 1);
    
    disp('初始化 PSO 粒子 (W矩阵)...');
    for i = 1:nPop
        % 稀疏初始化
        pop(i).Position = zeros(VarSize);
        % 只在有效位置随机生成一些非零值 (稀疏度 20% 左右)
        num_active = floor(length(valid_indices) * 0.2);
        active_idx = valid_indices(randperm(length(valid_indices), num_active));
        pop(i).Position(active_idx) = rand(1, num_active);
        
        pop(i).Velocity = zeros(VarSize);
        
        % 计算 MOSSR 适应度
        pop(i).Cost = CostFunction(pop(i).Position);
        
        pop(i).Best.Position = pop(i).Position;
        pop(i).Best.Cost = pop(i).Cost;
    end
    
    % 建立档案 (使用你提供的 DetermineDomination)
    pop = DetermineDomination(pop);
    rep = pop(~[pop.IsDominated]);
    rep = GridIndex(rep, nGrid); % 使用你提供的 GridIndex
    
    %% 4.3 PSO 主迭代
    disp('开始 PSO 优化权重矩阵 W...');
    for it = 1:MaxT
        % 计算当前迭代的变异概率 pm (参考第一段代码逻辑)
        % pm 随迭代次数增加而减小，初期探索，后期收敛
        pm = (1-(it-1)/(MaxT-1))^(1/mu); 
        
        for i = 1:nPop
            % --- 1. 标准 PSO 更新逻辑 (速度与位置) ---
            leader = SelectLeader(rep);
            pop(i).Velocity = w * pop(i).Velocity ...
                + c1 * rand(VarSize) .* (pop(i).Best.Position - pop(i).Position) ...
                + c2 * rand(VarSize) .* (leader.Position - pop(i).Position);
            
            pop(i).Velocity = max(pop(i).Velocity, -maxrate*L);
            pop(i).Velocity = min(pop(i).Velocity, maxrate*L);
            
            pop(i).Position = pop(i).Position + pop(i).Velocity;
            
            % --- 2. 边界处理与物理屏蔽 (Masking) ---
            pop(i).Position = max(pop(i).Position, VarMin);
            pop(i).Position = min(pop(i).Position, VarMax);
            pop(i).Position(~mask_flat) = 0; % 关键：确保噪声波段不参与
            
            % 更新基础适应度
            pop(i).Cost = CostFunction(pop(i).Position);
            
            % --- 3. 新增：变异机制 (针对权重矩阵 W) ---
            if rand < pm
                % 随机选择约 1% 的非噪声维度进行变异
                nMutate = max(1, round(0.01 * length(valid_indices))); 
                mutate_sub_idx = valid_indices(randperm(length(valid_indices), nMutate));
                
                % 对选中维度赋予新的随机权重 [0, 1]
                pop(i).Position(mutate_sub_idx) = rand(1, nMutate);
                
                % 变异后重新计算适应度
                pop(i).Cost = CostFunction(pop(i).Position);
                
                % 注意：这里采用 RoD (Replace or Dominate) 逻辑或简单更新
                % 为保持 PSO 稳定性，我们直接更新当前位置，并在下方判断 PBest
            end
            
            % --- 4. 更新个体最优 PBest ---
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
        
        % 7. 维护档案 (使用你提供的逻辑)
        rep = [rep; pop]; 
        rep = DetermineDomination(rep);
        rep = rep(~[rep.IsDominated]);
        rep = GridIndex(rep, nGrid);
        
        % 删除多余档案成员
        if numel(rep) > nRep
            % 这里需要调用 DeleteRepMemebr 和 WSIS
            % 为了代码独立性，我将相关逻辑简化集成，或者你可以确保 WSIS.m 存在
            try
                seq = WSIS(rep);
                Extra = numel(rep) - nRep;
                for e = 1:Extra
                    rep = DeleteRepMemebr(rep, seq);
                end
            catch
                % 如果没有 WSIS，随机删除
                Extra = numel(rep) - nRep;
                rep(randperm(numel(rep), Extra)) = [];
            end
        end
        
        % 衰减惯性
        w = w * 0.99;
    end
    
    %% 4.4 波段选择 (MOSSR 逻辑：基于 W 投票)
    disp('优化完成，执行 MOSSR 波段选择...');
    
    % 将档案中的所有 W 取出来
    x_rep = vertcat(rep.Position);
    f_rep = vertcat(rep.Cost);
    
    % 调用 MOSSR 的投票函数
    final_selected_bands = select_bands_test2(x_rep, f_rep, L, K);
    selected_bands_all{runIdx} = final_selected_bands;
    
    %% 4.5 CEM 验证 (保持不变)
    img_sel = img(:, final_selected_bands);
    d_sel = d(final_selected_bands);
    
    detectmap = reshape(detector(img_sel, d_sel', detector_Name), W, H);
    
    % 最后一轮画图
    if runIdx == nRuns
        figure(1); imagesc(detectmap); 
        title(['第', num2str(runIdx), '轮检测图 (PSO-MOSSR)']); 
        colormap jet; colorbar;
    end
    
    % AUC
    [AUC, ~] = cal_AUC(detectmap(:), img_gt(:), 1, 1);
    
    AUC_PFPD(runIdx) = AUC.PFPD;
    AUC_tauPD(runIdx) = AUC.tauPD;
    AUC_tauPF(runIdx) = AUC.tauPF;
    
    fprintf('  AUC (PF-PD): %.6f\n', AUC.PFPD);
end

%% 5. 结果汇总
fprintf('\n==================== 最终统计 ====================\n');
fprintf('平均 AUC (PF-PD): %.6f (Std: %.6f)\n', mean(AUC_PFPD), std(AUC_PFPD));
fprintf('最大 AUC (PF-PD): %.6f\n', max(AUC_PFPD));


