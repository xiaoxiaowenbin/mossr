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
c1_start = 1.0;  c1_end = 1.0; 
c2_start = 1.0;  c2_end = 2.0;
nGrid = 4;                   % 网格数 (用于 SelectLeader)
maxrate = 0.2;               % 速度限制
mu = 0.1;         % mutation  probability regulator

% ================= 文件名配置 =================
data_filename = 'WHU-Hi-River.hdr';
gt_filename   = 'target_mask.hdr';
% =============================================

%% 2. 数据加载与 MOSSR 预处理 (适配 WHU-Hi-River)
disp('正在加载 WHU-Hi-River 数据...');

% --- 2.1 读取影像 ---
if exist(data_filename, 'file')
    [img_src, ~] = enviread(data_filename); % 调用底部的读取函数
else
    error(['找不到文件: ' data_filename]);
end

% --- 2.2 读取真值 ---
if exist(gt_filename, 'file')
    [img_gt_raw, ~] = enviread(gt_filename);
    % 处理真值图维度 (如果是3维取第一层)
    if ndims(img_gt_raw) == 3
        img_gt_raw = img_gt_raw(:,:,1);
    end
    img_gt = double(img_gt_raw);
    img_gt(img_gt > 0) = 1; % 二值化
else
    error(['找不到真值: ' gt_filename]);
end

% --- 2.3 尺寸与归一化 ---
[W, H, L] = size(img_src);
img_2d = reshape(img_src, W * H, L);

% --- [新增] 预计算 SNR ---
disp('正在估算各波段 SNR...');
snr_vec = estimate_hsi_snr(img_src);

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
    w = 0.5;
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
    noise_bands = [1:1, (L-0):L];
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
        num_active = randi([floor(0.05*L), floor(0.4*L)]); 
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
    
    %% 4.3 升级版优化主循环 (对标 MOBS-TD 逻辑，集成 RoD)
    for it = 1:MaxT
        current_c1 = c1_start - (c1_start - c1_end) * (it/MaxT);
        current_c2 = c2_start + (c2_end - c2_start) * (it/MaxT);

        % 动态调整概率
        %pm = (1-(it-1)/(MaxT-1))^(1/mu); 
        pm = 0.5 * (1 - it/MaxT) + 0.05;
        %pc = (1-(it-1)/(MaxT-1))^(1/mu);
        pc =pm ;
        
        % 交叉后代容器初始化
        popc = []; 
        
        % --- A. 粒子更新与 RoD 变异 (在种群循环内) ---
        for i = 1:nPop
            % 1. 标准 PSO 更新
            leader = SelectLeader(rep);
            pop(i).Velocity = w * pop(i).Velocity ...
                + current_c1 * rand(VarSize) .* (pop(i).Best.Position - pop(i).Position) ...
                + current_c2 * rand(VarSize) .* (leader.Position - pop(i).Position);
            
            pop(i).Velocity = max(min(pop(i).Velocity, maxrate), -maxrate);
            
            % 创建一个临时解 NewSol 用于存储 PSO 移动后的位置
            NewSol = pop(i); 
            NewSol.Position = NewSol.Position + pop(i).Velocity;
            
            % 边界处理与掩膜屏蔽
            NewSol.Position = max(min(NewSol.Position, VarMax), VarMin);
            NewSol.Position(~mask_flat) = 0;

            % 【新增调优】：硬阈值截断，抹除极小权重，保持矩阵稀疏！
            % NewSol.Position(NewSol.Position < 0.05) = 0; 

            NewSol.Cost = CostFunction(NewSol.Position);
            
            % 使用 RoD 判定 PSO 移动：更新当前位置并同步更新 PBest
            pop(i) = RoD(NewSol, pop(i));
            
            % 2. 升级版 RoD 变异机制
            if rand < pm
                MutantSol = pop(i); % 基于当前位置进行变异
                
                % --- 核心逻辑：单点变异 ---
                j_idx = randi(length(valid_indices)); 
                j = valid_indices(j_idx); 
                dx = pm * (VarMax - VarMin); 
                
                lb = max(VarMin, MutantSol.Position(j) - dx);
                ub = min(VarMax, MutantSol.Position(j) + dx);
                
                MutantSol.Position(j) = unifrnd(lb, ub); 
                MutantSol.Position(~mask_flat) = 0; % 物理屏蔽
                MutantSol.Cost = CostFunction(MutantSol.Position);
                
                % 使用 RoD 判定变异：决定是否采纳变异并同步更新 PBest
                pop(i) = RoD(MutantSol, pop(i));
            end
        end
        
        % --- B. 档案交叉演化 (逻辑保持，但延后统一维护) ---
        if numel(rep) > 2 && rand < pc
            nC = 2 * floor(pc * numel(rep) / 2);
            c_idx = reshape(randperm(numel(rep), nC), nC/2, 2);
            popc = repmat(empty_particle, nC/2, 1);
            
            for k = 1:nC/2
                p1 = rep(c_idx(k,1)); p2 = rep(c_idx(k,2));
                alpha = rand(VarSize);
                popc(k).Position = alpha .* p1.Position + (1-alpha) .* p2.Position;
                popc(k).Position(~mask_flat) = 0;
                popc(k).Cost = CostFunction(popc(k).Position);
            end
        end
        
        % --- C. 统一维护档案 (模仿 MOBS-TD 汇总更新) ---
        % 将旧档案、当前种群、交叉后代汇总，进行非支配排序和 WSIS 筛选
        rep = MaintainArchive([rep; pop; popc], nRep, nGrid);
        if numel(rep) < 5 
            random_indices = randperm(nPop, min(nPop, 5));
            rep = [rep; pop(random_indices)];
            rep = DetermineDomination(rep); % 重新排一次
            rep = rep(~[rep.IsDominated]);
        end
        fprintf('Iter %d/%d: rep数: %d\n', it, MaxT, numel(rep));
        
        w = w * 0.99; % 惯性权重衰减
    end
    

    
    %% 4.4 波段选择 (MOSSR 逻辑：基于 W 投票)
    disp('优化完成，执行 MOSSR 波段选择...');
    
    % 将档案中的所有 W 取出来
    x_rep = vertcat(rep.Position);
    f_rep = vertcat(rep.Cost);
    
    % 调用 MOSSR 的投票函数
    final_selected_bands = select_bands_test3(x_rep, f_rep, L, K, snr_vec, d);
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

