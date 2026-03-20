clc;clear;close all;
%{ 
    Reference: MOBS-TD Algorithm
    Dataset: San Diego (Cropped to 100x100, 189 Bands)
%}

%% General Setup
nRuns = 10;       % 独立运行次数

%% Parameters (Algorithm specific)
MaxT = 100;        % Maximum number of iterations
nPop = 100;        % population size
nRep = 20;         % Candidate solution set size

% PSO Parameters
w_start = 0.5;     % Initial inertia factor
wdamp = 0.99;      % Attenuation factor
c1 = 1;            % Global learning factor
c2 = 1;            % Individual learning factor

nGrid = 4;         % Number of grids per dimension
mu = 0.1;          % Mutation probability
maxrate = 0.2;     % Speed control

seed_base = 123;      % 基础随机种子，修改此值可改变整套实验的随机序列

rng(seed_base, 'twister');

%% Load Data & Pre-processing (San Diego Specific)
disp('===========================================================');
disp('             LOADING SAN DIEGO DATASET                     ');
disp('===========================================================');

% --- 1. 加载文件 ---
if exist('Sandiego.mat', 'file') && exist('PlaneGT.mat', 'file')
    load Sandiego.mat; 
    load PlaneGT.mat;
    disp('数据文件加载成功。');
else
    error('请确保当前路径下存在 Sandiego.mat 和 PlaneGT.mat');
end

% --- 2. 空间裁剪 (Spatial Cropping) ---
% 原始 Sandiego 是 400x400，GT 是 100x100，飞机在左上角
[H_full, W_full, L_full] = size(Sandiego);
[H_gt, W_gt] = size(PlaneGT);
img_gt = PlaneGT;

if H_full ~= H_gt || W_full ~= W_gt
    fprintf('尺寸调整: 从 %dx%d 裁剪至 %dx%d ...\n', H_full, W_full, H_gt, W_gt);
    img_src = Sandiego(1:H_gt, 1:W_gt, :);
else
    img_src = Sandiego;
end
[W, H, L] = size(img_src);

% --- 3. 剔除坏波段 (Spectral Pruning) ---
if L >= 224
    disp('正在剔除坏波段 (保留 189 个有效波段)...');
    % San Diego 标准剔除列表 (去除水汽吸收带等)
    bad_bands = [1:6, 33:35, 97, 107:113, 153:166, 221:224];
    good_bands_idx = setdiff(1:224, bad_bands);
    
    % 更新图像数据
    img_src = img_src(:, :, good_bands_idx);
    
    % 更新波段数 L
    [W, H, L] = size(img_src); 
    fprintf('当前有效波段数: %d\n', L);
else
    warning('原始波段数不足224，跳过剔除步骤，当前波段: %d', L);
end

% --- 4. 数据整形与归一化 ---
N_pixels = W * H;
img = reshape(img_src, N_pixels, L); % 这里的 img 对应之前代码的 reshaped 数据

% 归一化 (Min-Max)
img = double(img);
min_val = min(img(:));
max_val = max(img(:));
img = (img - min_val) ./ (max_val - min_val + eps);

% --- 5. 提取目标光谱与计算特征 ---
disp('计算目标光谱及图像特征 (Entropy, Spatial-Spectral)...');
d = get_target(img, img_gt);

%% ========== VD估计（确定K值）==========
% 这一步决定选多少个波段
t_vd = 1e-4; 
fprintf('执行 VD 估计 (HFC/NWHFC)...\n');
hfc_vd = HFC1(img_src, t_vd);
nwhfc_vd = NWHFC(img_src, t_vd);
nVar = 2 * max(hfc_vd, nwhfc_vd); % 通常取 HFC 和 NWHFC 较大值的两倍作为冗余
fprintf('最终选择的波段数量 (K): %d\n', nVar);

% 这些计算比较耗时，放在循环外只计算一次
En = Entrop(img); 
D = spectral_spatial(img);

%% Cost Function Definition
CostFunction = @(x,h,d,m,t) BS_model(x,h,d,m,t);

%% Initialization of Storage for 10 Runs
RunResults = repmat(struct('Solution', [], 'AUC_PFPD', 0, 'AUC_tauPD', 0, 'AUC_tauPF', 0), nRuns, 1);

%% Main Loop for Independent Runs
for run = 1:nRuns
    current_seed = seed_base + run - 1;
    rng(current_seed, 'twister');
    
    fprintf('-------------------------------------------\n');
    fprintf('Starting Run %d / %d (Seed: %d)\n', run, nRuns, current_seed);
    fprintf('-------------------------------------------\n');
    %% Initialization for Current Run
    VarSize = [1 nVar];                            
    VarMin = 1;                                    
    VarMax = L; % 注意：这里的 L 已经是剔除波段后的 189 了
    
    w = w_start; 

    % Initialize Empty Particle
    empty_particle.Position = [];                  
    empty_particle.Velocity = [];                  
    empty_particle.Cost = [];                      
    empty_particle.Best.Position = [];             
    empty_particle.Best.Cost = [];                 
    empty_particle.IsDominated = [];               
    empty_particle.GridIndex = [];                 
    empty_particle.GridSubIndex = [];              
    
    pop = repmat(empty_particle, nPop, 1);         

    % 1st-generation population
    for i = 1:nPop 
        pop(i).Position = sort(randperm(VarMax, nVar));
        pop(i).Velocity = zeros(VarSize);
        % 注意：这里 img' 是转置，符合 BS_model 的输入要求 (L * N)
        pop(i).Cost = CostFunction(pop(i).Position, En, D, img', d); 
        pop(i).Best.Position = pop(i).Position;
        pop(i).Best.Cost = pop(i).Cost;
    end

    % Determine Domination
    pop = DetermineDomination(pop);
    rep = pop(~[pop.IsDominated]);

    % Grid
    rep = GridIndex(rep, nGrid);

    %% Optimization Loop
    for it = 1:MaxT
        
        for i = 1:nPop
            leader = SelectLeader(rep);
            
            pop(i).Velocity = w*pop(i).Velocity ...
                +c1*rand(VarSize).*(pop(i).Best.Position-pop(i).Position) ...
                +c2*rand(VarSize).*(leader.Position-pop(i).Position);
            
            pop(i).Velocity = max(pop(i).Velocity, (-1)*maxrate*VarMax);
            pop(i).Velocity = min(pop(i).Velocity, maxrate*VarMax);
            pop(i).Velocity = fix(pop(i).Velocity);
            
            pop(i).Position = pop(i).Position + pop(i).Velocity;
            pop(i).Position = limitPositionVariables(pop(i).Position, VarMin, VarMax);
            
            pop(i).Cost = CostFunction(pop(i).Position, En, D, img', d);
            
            %% Mutation
            pm = (1-(it-1)/(MaxT-1))^(1/mu);
            if rand < pm
                NewSol.Position = Mutate(pop(i).Position, pm, VarMin, VarMax);
                NewSol.Position = limitPositionVariables(NewSol.Position,VarMin,VarMax);
                NewSol.Cost = CostFunction(NewSol.Position, En, D, img', d);
                pop(i) = RoD(NewSol,pop(i));
            end
        end

        %% Update Rep Set
        pop = DetermineDomination(pop);
        rep = [rep
             pop(~[pop.IsDominated])];
        rep = DetermineDomination(rep);    
        rep = rep(~[rep.IsDominated]);
        
        %% Crossover
        pc = (1-(it-1)/(MaxT-1))^(1/mu);
        num_rep = numel(rep);
        if rand < pc
            nCrossover = 2*floor(pc*num_rep/2);
            popc = repmat(empty_particle, nCrossover/2, 1); 
            cross_index = reshape(randperm(num_rep,nCrossover),nCrossover/2,2);
            for k = 1:nCrossover/2
                p1 = rep(cross_index(k,1));
                p2 = rep(cross_index(k,2));
                
                popc(k).Position = Crossover(p1.Position, p2.Position, En); 
                popc(k).Velocity = ((p1.Velocity + p2.Velocity)*sqrt(dot(p1.Velocity,p1.Velocity))) ...
                    /((sqrt(dot(p1.Velocity,p1.Velocity))+sqrt(dot(p2.Velocity,p2.Velocity)))+inf);
                popc(k).Velocity = max(popc(k).Velocity, (-1)*maxrate*VarMax);
                popc(k).Velocity = min(popc(k).Velocity, maxrate*VarMax);
                popc(k).Velocity = fix(popc(k).Velocity);
                
                popc(k).Cost = CostFunction(popc(k).Position, En, D, img', d);
            end
            rep = [rep; popc];
            rep = DetermineDomination(rep);
            rep = rep(~[rep.IsDominated]);
        end
        
        rep = GridIndex(rep, nGrid);

        if numel(rep) > nRep
            Extra = numel(rep) - nRep;
            seq = WSIS(rep);
            for e = 1:Extra
                rep = DeleteRepMemebr(rep, seq);
            end        
        end

        % Plot Costs
        if mod(it, 50) == 0 || it == 1
            figure(1);
            PlotCosts(pop, rep);
            title(['Run: ' num2str(run) ' | Iter: ' num2str(it)]);
            pause(0.01);
        end
        
        w = w * wdamp;
    end
    
    %% Run Evaluation
    disp(['Run ' num2str(run) ' Calculating AUC...']);
    
    repSet = {rep.Position};
    detector_Name = 'CEM';
    
    fSolution = MSR(repSet, detector_Name, img, W, H, d);
    
    % Detection Map
    detectmap = reshape(detector(img(:,fSolution), d(fSolution)', detector_Name), W, H);
    
    % AUC Calculation
    det_map_vec = detectmap(:);
    GT_vec = img_gt(:);
    [AUC, ~] = cal_AUC(det_map_vec, GT_vec, 1, 1);
    
    RunResults(run).Solution = fSolution;
    RunResults(run).AUC_PFPD = AUC.PFPD;
    RunResults(run).AUC_tauPD = AUC.tauPD;
    RunResults(run).AUC_tauPF = AUC.tauPF;
    
    disp(['Run ' num2str(run) ' AUC(PF-PD): ' num2str(AUC.PFPD)]);
end

%% Statistical Analysis
disp('===========================================================');
disp('                  FINAL STATISTICS (San Diego)             ');
disp('===========================================================');

all_AUCs = [RunResults.AUC_PFPD];
mean_AUC = mean(all_AUCs);
std_AUC = std(all_AUCs);
best_AUC = max(all_AUCs);
[~, best_run_idx] = max(all_AUCs);

fprintf('Mean AUC (PF-PD): %.6f\n', mean_AUC);
fprintf('Std  AUC (PF-PD): %.6f\n', std_AUC);
fprintf('Best AUC (PF-PD): %.6f (Run %d)\n', best_AUC, best_run_idx);

%% Visualization of Best Run
best_Solution = RunResults(best_run_idx).Solution;
detectmap_best = reshape(detector(img(:,best_Solution), d(best_Solution)', detector_Name), W, H);

figure('Name', 'Best Result San Diego');
subplot(1,2,1); imagesc(img_gt); title('Ground Truth'); axis image;
subplot(1,2,2); imagesc(detectmap_best); title(['Best Detection (AUC: ' num2str(best_AUC) ')']); axis image; colorbar;

% 显示原始波段索引（需要映射回剔除前的索引吗？通常论文中如果提到了剔除，就直接报新索引，或者注明是Reduced Space）
disp(['Optimal band subset (Indices in Reduced 189 bands): (' num2str(best_Solution) ')']);

% 如果你想知道这些波段对应原始224个波段里的第几个：
% original_indices = good_bands_idx(best_Solution);
% disp(['Optimal band subset (Original 224 Indices): (' num2str(original_indices) ')']);