%% Enhanced MOPSO-MOSSR Algorithm (Hybrid Version)
tic; clear; clc; close all;

%% 1. 核心配置与参数融合
nRuns = 10;
seed_base = 123;
detector_Name = 'CEM';

% 算法超参数 (融合代码2的标准约束)
MaxT = 100; nPop = 100; nRep = 20;
w = 0.5; wdamp = 0.99; c1 = 1.0; c2 = 1.0;
nGrid = 4; mu = 0.1; maxrate = 0.1; 

%% 2. 数据预处理 (保留 MOSSR 核心)
load abu-airport-2.mat; 
[W, H, L] = size(data);
img_2d = reshape(data, W*H, L);
% 归一化
img = (img_2d - min(img_2d)) ./ (max(img_2d) - min(img_2d) + eps);
d = get_target(img, map);
if size(d,1) > size(d,2), d = d'; end
YtY = img' * img;

% VD 估计确定 K
K = 2 * max(HFC1(data, 1e-4), NWHFC(data, 1e-4));
fprintf('目标波段数 K = %d\n', K);

%% 3. 噪声屏蔽与空间定义 (代码1的优点)
num_vars = L * L;
% 物理屏蔽：排除光谱两端的噪声水汽波段
noise_bands = [1:8, (L-5):L];
mask_matrix = ones(L, L);
mask_matrix(noise_bands, :) = 0; mask_matrix(:, noise_bands) = 0;
mask_flat = reshape(mask_matrix, 1, num_vars);
valid_indices = find(mask_flat == 1);

%% 4. 主循环 (融合 PSO 逻辑)
for runIdx = 1:nRuns
    rng(seed_base + runIdx - 1, 'twister');
    
    % 初始化：采用代码1的稀疏初始化 (20% 激活)
    pop = repmat(struct('Position',[],'Velocity',[],'Cost',[],'Best',struct('Position',[],'Cost',[]),...
                 'IsDominated',[],'GridIndex',[],'GridSubIndex',[]), nPop, 1);
    
    for i = 1:nPop
        pop(i).Position = zeros(1, num_vars);
        num_active = floor(length(valid_indices) * 0.2);
        active_idx = valid_indices(randperm(length(valid_indices), num_active));
        pop(i).Position(active_idx) = rand(1, num_active);
        pop(i).Velocity = zeros(1, num_vars);
        pop(i).Cost = mossr_fitness_test2(pop(i).Position, YtY, L, d);
        pop(i).Best.Position = pop(i).Position;
        pop(i).Best.Cost = pop(i).Cost;
    end
    
    % 初始档案建立
    pop = DetermineDomination(pop);
    rep = pop(~[pop.IsDominated]);
    rep = GridIndex(rep, nGrid);
    
    % --- 迭代开始 ---
    for it = 1:MaxT
        % 动态变异概率
        pm = (1-(it-1)/(MaxT-1))^(1/mu);
        
        for i = 1:nPop
            leader = SelectLeader(rep);
            % 速度更新 (代码2的标准公式)
            pop(i).Velocity = w * pop(i).Velocity ...
                + c1 * rand(1,num_vars) .* (pop(i).Best.Position - pop(i).Position) ...
                + c2 * rand(1,num_vars) .* (leader.Position - pop(i).Position);
            
            % 边界限制 + 噪声掩膜 (融合)
            pop(i).Position = pop(i).Position + pop(i).Velocity;
            pop(i).Position = max(min(pop(i).Position, 1), 0);
            pop(i).Position(~mask_flat) = 0; % 强制物理屏蔽
            
            % 变异逻辑 (代码1的有效维度变异)
            if rand < pm
                nMutate = max(1, round(0.01 * length(valid_indices)));
                pop(i).Position(valid_indices(randperm(length(valid_indices), nMutate))) = rand(1, nMutate);
            end
            
            pop(i).Cost = mossr_fitness_test2(pop(i).Position, YtY, L, d);
            
            % 更新 PBest (Dominates 判断)
            if Dominates(pop(i).Cost, pop(i).Best.Cost)
                pop(i).Best.Position = pop(i).Position;
                pop(i).Best.Cost = pop(i).Cost;
            end
        end
        
        % 档案维护 (带容错的 WSIS)
        rep = [rep; pop(~[pop.IsDominated])];
        rep = DetermineDomination(rep);
        rep = rep(~[rep.IsDominated]);
        rep = GridIndex(rep, nGrid);
        if numel(rep) > nRep
            try
                seq = WSIS(rep);
                for e = 1:(numel(rep)-nRep), rep = DeleteRepMemebr(rep, seq); end
            catch
                rep(randperm(numel(rep), numel(rep)-nRep)) = [];
            end
        end
        w = w * wdamp;
    end
    
    % 结果提取与检测 (代码2的全指标记录)
    x_pareto = vertcat(rep.Position);
    f_pareto = vertcat(rep.Cost);
    fSolution = select_bands_test2(x_pareto, f_pareto, L, K);
    
    res_map = reshape(detector(img(:,fSolution), d(fSolution)', detector_Name), W, H);
    [AUC, AUCnor] = cal_AUC(res_map(:), map(:), 1, 1);
    
    AUC_PFPD(runIdx) = AUC.PFPD;
    AUCnor_PFPD(runIdx) = AUCnor.PFPD;
    fprintf('Run %d: AUC = %.6f\n', runIdx, AUC.PFPD);
end

%% 5. 统计输出
fprintf('\nFinal Mean AUC: %.6f, Mean AUCnor: %.6f\n', mean(AUC_PFPD), mean(AUCnor_PFPD));