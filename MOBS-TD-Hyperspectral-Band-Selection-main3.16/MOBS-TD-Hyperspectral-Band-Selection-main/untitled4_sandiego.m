%% Main Script: MOPSO-based Band Selection (MOBS-TD Logic - Fixed)
%  PSO 代码复现，包含 10 轮循环和随机种子控制
clc; clear; close all;

%% 1. 全局配置与数据加载
nRuns = 10;                  % 总运行轮数
seed_base = 123;             % 基础随机种子
target_class_id = 1;
detector_Name = 'CEM';       % 检测器类型

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

% --- 数据整形与归一化 ---
img_reshaped = reshape(double(img_src), W * H, L);
min_val = min(img_reshaped(:));
max_val = max(img_reshaped(:));
img_reshaped = (img_reshaped - min_val) ./ (max_val - min_val + eps);

% 提取目标
d = get_target(img_reshaped, img_gt);

% 预计算相关矩阵 Y'Y
disp('预计算相关矩阵 Y''Y...');
YtY = img_reshaped' * img_reshaped;
num_bands = L;

% === VD 估计确定波段数 ===
t_vd = 1e-4; 
fprintf('执行 VD 估计 (HFC/NWHFC)...\n');
hfc_vd = HFC1(img_src, t_vd);
nwhfc_vd = NWHFC(img_src, t_vd);
K = 2 * max(hfc_vd, nwhfc_vd); % 最终要选出的波段数
fprintf('最终选择的波段数量 (K): %d\n', K);

%% --- PSO 核心参数 (针对 MOSSR 连续优化调整) ---
% MOSSR 优化的是系数矩阵 (L*L)，维度极高，需要连续变量优化
num_vars = num_bands * num_bands; 
VarSize = [1, num_vars];   % [1, 26244]
VarMin = 0;                % 系数最小 0
VarMax = 1;                % 系数最大 1

MaxT = 100;        % 迭代次数 (维度太高，太多跑不动，建议先设少测试)
nPop = 100;        % 种群大小
nRep = 20;        % 档案大小
w = 0.5;          % 惯性权重
wdamp = 0.99;     % 阻尼系数
c1 = 1.0;         % 个体学习因子
c2 = 1.0;         % 全局学习因子
nGrid = 4;        % 网格数
mu = 0.1;         % 变异概率
maxrate = 0.1;    % 速度限制因子 (相对于 VarMax-VarMin)

VelMax = maxrate * (VarMax - VarMin);
VelMin = -VelMax;

%% 2. 初始化结果存储
AUC_PFPD = zeros(nRuns, 1);
AUC_tauPD = zeros(nRuns, 1);
AUC_tauPF = zeros(nRuns, 1);
AUCnor_PFPD = zeros(nRuns, 1);
selected_bands_all = cell(nRuns, 1);

%% 3. 多轮 MOPSO 优化循环
for runIdx = 1:nRuns
    fprintf('\n==================== 第 %d/%d 轮运行 (MOPSO) ====================\n', runIdx, nRuns);
    
    % 设置随机种子
    current_seed = seed_base + runIdx - 1;
    rng(current_seed, 'twister');
    fprintf('本轮随机种子: %d\n', current_seed);
    
    % --- 初始化种群结构体 ---
    empty_particle.Position = [];
    empty_particle.Velocity = [];
    empty_particle.Cost = [];
    empty_particle.Best.Position = [];
    empty_particle.Best.Cost = [];
    empty_particle.IsDominated = [];
    empty_particle.GridIndex = [];
    empty_particle.GridSubIndex = [];
    
    pop = repmat(empty_particle, nPop, 1);
    
    disp('初始化种群 (这可能需要一点时间)...');
    for i = 1:nPop
        % [修正] 必须是连续变量初始化，大小为 num_vars
        pop(i).Position = unifrnd(VarMin, VarMax, VarSize); 
        pop(i).Velocity = zeros(VarSize);
        
        % 计算适应度
        pop(i).Cost = mossr_fitness_test2(pop(i).Position, YtY, num_bands, d);
        
        % 更新个人最优
        pop(i).Best.Position = pop(i).Position;
        pop(i).Best.Cost = pop(i).Cost;
    end
    
    % 确定支配关系并建立档案 (确保你有 DetermineDomination 和 GridIndex 函数)
    pop = DetermineDomination(pop);
    rep = pop(~[pop.IsDominated]);
    rep = GridIndex(rep, nGrid);
    
    %% --- MOPSO 主迭代 ---
    disp('开始 MOPSO 迭代...');
    
    for it = 1:MaxT
        for i = 1:nPop
            % 选择 Leader
            leader = SelectLeader(rep);
            
            % 更新速度
            pop(i).Velocity = w * pop(i).Velocity ...
                + c1 * rand(VarSize) .* (pop(i).Best.Position - pop(i).Position) ...
                + c2 * rand(VarSize) .* (leader.Position - pop(i).Position);
            
            % [修正] 速度限制 (去除 fix，这是连续优化)
            pop(i).Velocity = max(pop(i).Velocity, VelMin);
            pop(i).Velocity = min(pop(i).Velocity, VelMax);
            
            % 更新位置
            pop(i).Position = pop(i).Position + pop(i).Velocity;
            
            % [修正] 边界限制 (Clip)
            pop(i).Position = max(pop(i).Position, VarMin);
            pop(i).Position = min(pop(i).Position, VarMax);
            
            % 计算适应度
            pop(i).Cost = mossr_fitness_test2(pop(i).Position, YtY, num_bands, d);
            
            % 变异操作 (简化版连续变异)
            % 注意：如果原来的 Mutate 函数是针对整数索引的，这里不能用，改用简单的随机扰动
            pm = (1-(it-1)/(MaxT-1))^(1/mu);
            if rand < pm
                % 随机选择少量维度进行变异，避免破坏整体结构
                nMutate = max(1, round(0.01 * num_vars)); % 变异 1% 的基因
                idx = randperm(num_vars, nMutate);
                pop(i).Position(idx) = unifrnd(VarMin, VarMax, [1, nMutate]);
                pop(i).Cost = mossr_fitness_test2(pop(i).Position, YtY, num_bands, d);
            end
            
            % 更新个人最优 (简单的支配判断)
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
        
        % 更新档案 (Repository)
        pop = DetermineDomination(pop);
        rep = [rep; pop(~[pop.IsDominated])];
        rep = DetermineDomination(rep);
        rep = rep(~[rep.IsDominated]);
        
        % 维护档案大小
        rep = GridIndex(rep, nGrid);
        if numel(rep) > nRep
            Extra = numel(rep) - nRep;
            % 简单的删除策略 (删除拥挤度高的)
            [~, idx] = sort([rep.GridIndex], 'descend'); 
            % 如果你有 DeleteRepMemebr 请使用，这里用简单删除代替以防报错
            rep(idx(1:Extra)) = []; 
        end
        
        % 更新惯性权重
        w = w * wdamp;
        
        if mod(it, 10) == 0
            fprintf('  Iter: %d, Rep Size: %d\n', it, numel(rep)); 
        end
    end
    
    disp(['第', num2str(runIdx), '轮优化完成！']);
    
    %% 4. 结果评估 (关键修正：构建 Pareto 矩阵供选择函数使用)
    
    % [修正] 将 MOPSO 的 rep 结构体转换为矩阵形式
    nRepFit = numel(rep);
    x_pareto = zeros(nRepFit, num_vars);
    f_pareto = zeros(nRepFit, 3); % 假设 fitness 返回 2 个目标
    
    for k = 1:nRepFit
        x_pareto(k, :) = rep(k).Position;
        f_pareto(k, :) = rep(k).Cost;
    end
    
    % 使用 MSR 从 Pareto 前沿中选出唯一最优解
    % 这里的 select_bands_test2 内部会进行 reshape 和 聚类
    fSolution = select_bands_test2(x_pareto, f_pareto, num_bands, K);
    
    % 保存选中的波段
    selected_bands_all{runIdx} = fSolution;
    
    % [修正] 增强容错性的打印方式
    fprintf('选定波段: ');
    fprintf('%d ', fSolution);
    fprintf('\n');
    
    % 执行检测 (使用选定波段)
    detectmap = reshape(detector(img_reshaped(:,fSolution), d(fSolution)', detector_Name), W, H);
    
    % 如果是最后一轮，画图
    if runIdx == nRuns
        figure(1); imagesc(detectmap); 
        title(['第', num2str(runIdx), '轮检测图 (MOPSO)']); 
        colormap jet; colorbar;
    end
    
    % 计算 AUC
    det_map_vec = detectmap(:);
    GT_vec = img_gt(:);
    [AUC, AUCnor] = cal_AUC(det_map_vec, GT_vec, 1, 1);
    
    % 记录结果
    AUC_PFPD(runIdx) = AUC.PFPD;
    AUC_tauPD(runIdx) = AUC.tauPD;
    AUC_tauPF(runIdx) = AUC.tauPF;
    AUCnor_PFPD(runIdx) = AUCnor.PFPD;
    AUCnor_tauPD(runIdx) = AUCnor.tauPD;
    AUCnor_tauPF(runIdx) = AUCnor.tauPF;
    
    fprintf('  AUC (PF-PD): %.6f\n', AUC.PFPD);
end

%% 5. 结果汇总
fprintf('\n==================== 10轮 MOPSO 统计结果 ====================\n');
fprintf('平均 AUC (PF-PD): %.6f (标准差: %.6f)\n', mean(AUC_PFPD), std(AUC_PFPD));
fprintf('平均 Normalized AUC: %.6f\n', mean(AUCnor_PFPD));

