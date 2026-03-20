clc;clear;close all;

%% ================= 配置部分 (用户设置) =================
nRuns = 10;           % 独立运行次数
seed_base = 123;      % 随机种子，保证可复现

% 【关键】文件名配置 (请确保这4个文件都在当前文件夹下)
data_filename = 'WHU-Hi-River.hdr';     % 原始影像头文件
gt_filename   = 'target_mask.hdr';      % 目标真值头文件

%% Parameters (算法参数)
MaxT = 100;        % 最大迭代次数
nPop = 100;        % 种群大小
nRep = 20;         % 档案大小

% PSO 参数
w_start = 0.5;     
wdamp = 0.99;      
c1 = 1;            
c2 = 1;            
nGrid = 4;         
mu = 0.1;          
maxrate = 0.2;     

%% ================= 1. 加载 WHU-Hi-River 数据 =================
% 设置随机种子 (用于预处理)
rng(seed_base, 'twister');

disp('Loading WHU-Hi-River data...');

% --- 1.1 读取高光谱图像 (使用 enviread) ---
if exist(data_filename, 'file')
    [img_src, info] = enviread(data_filename);
    disp(['Image loaded. Size: ', num2str(size(img_src))]);
else
    error(['找不到影像文件: ' data_filename '，请检查是否已将文件复制到当前目录。']);
end

% --- 1.2 读取真值 (Ground Truth) (改用 enviread) ---
% WHU数据集的GT通常也是ENVI格式
if exist(gt_filename, 'file')
    [img_gt_raw, gt_info] = enviread(gt_filename);
    
    % WHU的mask通常是二维矩阵，如果是三维(1波段)，转换一下
    if ndims(img_gt_raw) == 3
        img_gt_raw = img_gt_raw(:,:,1);
    end
    
    % 转换为 double 并二值化 (0是背景，非0是目标)
    img_gt = double(img_gt_raw);
    img_gt(img_gt > 0) = 1; 
    
    if sum(img_gt(:)) == 0
        error('真值图全为0 (无目标)，请检查 target_mask 读取是否正确。');
    else
        fprintf('真值加载成功，目标像素数: %d\n', sum(img_gt(:)));
    end
else
    error(['找不到真值文件: ' gt_filename '，请从 Signature-based detection 文件夹复制 target_mask 和 target_mask.hdr']);
end

%% ================= 2. 数据预处理 =================
disp('Pre-processing data...');

% 获取尺寸
[W, H, L] = size(img_src);

% 检查尺寸匹配
[Wg, Hg] = size(img_gt);
if W ~= Wg || H ~= Hg
    error('影像尺寸 (%dx%d) 与 真值尺寸 (%dx%d) 不匹配！', W, H, Wg, Hg);
end

% 归一化 (转换为 double 并映射到 0-1)
img_src = double(img_src);
min_val = min(img_src(:));
max_val = max(img_src(:));
img_src = (img_src - min_val) ./ (max_val - min_val + eps);

% Reshape 为 (Pixels x Bands)
img = reshape(img_src, W * H, L);

% 提取目标光谱特征 (使用加载的 img_gt)
d = get_target(img, img_gt);

% 计算耗时特征 (只算一次)
disp('Calculating Entropy and Spatial-Spectral matrix...');
En = Entrop(img); 
D = spectral_spatial(img);

%% ========== 3. VD估计（确定K值）==========
t_vd = 1e-4; 
fprintf('执行 VD 估计 (HFC/NWHFC)...\n');
hfc_vd = HFC1(img_src, t_vd);
nwhfc_vd = NWHFC(img_src, t_vd);
% 动态确定波段选择数量
nVar = 2 * max(hfc_vd, nwhfc_vd); 
% 边界检查：WHU-River波段较少，防止溢出
nVar = max(min(nVar, floor(L/2)), 3); 
fprintf('最终选择的波段数量 (K): %d\n', nVar);

%% Cost Function Definition
CostFunction = @(x,h,d,m,t) BS_model(x,h,d,m,t);

%% Initialization of Storage for 10 Runs
RunResults = repmat(struct('Solution', [], 'AUC_PFPD', 0, 'AUC_tauPD', 0, 'AUC_tauPF', 0), nRuns, 1);

%% ================= 4. 主循环 (10 Runs) =================
for run = 1:nRuns
    w_start = 0.5;
    % 设置每轮独立的随机种子
    current_seed = seed_base + run - 1;
    rng(current_seed, 'twister');

    fprintf('-------------------------------------------\n');
    fprintf('Starting Run %d / %d (Seed: %d)\n', run, nRuns, current_seed);
    fprintf('-------------------------------------------\n');
    
    %% Initialization for Current Run
    VarSize = [1 nVar];                            
    VarMin = 1;                                    
    VarMax = L;                                    
    
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

    % 1st-generation
    for i = 1:nPop 
        pop(i).Position = sort(randperm(VarMax, nVar));
        pop(i).Velocity = zeros(VarSize);
        pop(i).Cost = CostFunction(pop(i).Position, En, D, img', d);
        pop(i).Best.Position = pop(i).Position;
        pop(i).Best.Cost = pop(i).Cost;
    end

    pop = DetermineDomination(pop);
    rep = pop(~[pop.IsDominated]);
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
            
            % Mutation
            pm = (1-(it-1)/(MaxT-1))^(1/mu);
            if rand < pm
                NewSol.Position = Mutate(pop(i).Position, pm, VarMin, VarMax);
                NewSol.Position = limitPositionVariables(NewSol.Position,VarMin,VarMax);
                NewSol.Cost = CostFunction(NewSol.Position, En, D, img', d);
                pop(i) = RoD(NewSol,pop(i));
            end
        end

        % Update Rep Set
        pop = DetermineDomination(pop);
        rep = [rep; pop(~[pop.IsDominated])];
        rep = DetermineDomination(rep);    
        rep = rep(~[rep.IsDominated]);
        
        % Crossover
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

        % Display
        if mod(it, 20) == 0 || it == 1
            figure(1);
            PlotCosts(pop, rep);
            title(['Run: ', num2str(run), ' | Iter: ', num2str(it), ' | Rep: ', num2str(numel(rep))]);
            pause(0.01);
        end
        
        w = w * wdamp;
    end
    
    %% Run Evaluation
    disp(['Run ' num2str(run) ' Calculating AUC...']);
    repSet = {rep.Position};
    detector_Name = 'CEM';
    
    fSolution = MSR(repSet, detector_Name, img, W, H, d);
    
    detectmap = reshape(detector(img(:,fSolution), d(fSolution)', detector_Name), W, H);
    
    det_map_vec = detectmap(:);
    GT_vec = img_gt(:);
    [AUC, ~] = cal_AUC(det_map_vec, GT_vec, 1, 1);
    
    RunResults(run).Solution = fSolution;
    RunResults(run).AUC_PFPD = AUC.PFPD;
    RunResults(run).AUC_tauPD = AUC.tauPD;
    RunResults(run).AUC_tauPF = AUC.tauPF;
    
    disp(['Run ' num2str(run) ' AUC(PF-PD): ' num2str(AUC.PFPD)]);
end

%% ================= 5. 统计与绘图 =================
disp('================ FINAL STATISTICS ================');
all_AUCs = [RunResults.AUC_PFPD];
mean_AUC = mean(all_AUCs);
std_AUC = std(all_AUCs);
[best_AUC, best_run_idx] = max(all_AUCs);
worst_AUC = min(all_AUCs);

fprintf('Mean AUC (PF-PD): %.6f\n', mean_AUC);
fprintf('Std  AUC (PF-PD): %.6f\n', std_AUC);
fprintf('Best AUC (PF-PD): %.6f (Run %d)\n', best_AUC, best_run_idx);

% 绘制最佳结果
best_Solution = RunResults(best_run_idx).Solution;
detectmap_best = reshape(detector(img(:,best_Solution), d(best_Solution)', detector_Name), W, H);

figure('Name', 'Best Result WHU-Hi-River');
subplot(1,2,1); imagesc(img_gt); title('Ground Truth (Mask)'); axis image;
subplot(1,2,2); imagesc(detectmap_best); title(['Detection Map (AUC: ' num2str(best_AUC) ')']); axis image; colorbar;
disp(['Optimal bands: (' num2str(best_Solution) ')']);

%% ================= 附：修复版 ENVIREAD 函数 =================
function [data, info] = enviread(filename)
% ENVIREAD 读取 ENVI 格式的高光谱图像或掩膜 (修复 data_type 解析错误)

    % --- 1. 定位 HDR 和 数据文件 ---
    [pathstr, name, ext] = fileparts(filename);
    if strcmp(ext, '.hdr')
        hdrfile = filename;
        datfile = fullfile(pathstr, name); 
        % 尝试寻找数据文件 (无后缀, .img, .dat, .bin, .raw)
        exts = {'', '.img', '.dat', '.bin', '.raw'};
        found = false;
        for k = 1:length(exts)
            temp_path = [datfile exts{k}];
            if exist(temp_path, 'file')
                datfile = temp_path;
                found = true;
                break;
            end
        end
        if ~found
             % 如果都没找到，尝试仅仅使用文件名
             datfile = fullfile(pathstr, name);
        end
    else
        hdrfile = [filename '.hdr'];
        datfile = filename;
    end

    % --- 2. 读取 HDR 头文件 ---
    fid = fopen(hdrfile, 'r');
    if fid == -1, error(['无法打开 HDR 文件: ' hdrfile]); end
    
    info = struct();
    while ~feof(fid)
        line = fgetl(fid);
        % 跳过空行或非键值对行
        if ischar(line) && contains(line, '=')
            parts = strsplit(line, '=');
            
            % --- 关键修复步骤开始 ---
            key = strtrim(parts{1}); % 去除首尾空格
            val = strtrim(parts{2}); % 去除首尾空格
            
            % 1. 转小写 (Data Type -> data type)
            key = lower(key);
            
            % 2. 将空格替换为下划线 (data type -> data_type)
            key = strrep(key, ' ', '_');
            
            % 3. 移除可能存在的非法字符 (如括号)
            key = strrep(key, '(', '');
            key = strrep(key, ')', '');
            % --- 关键修复步骤结束 ---
            
            info.(key) = val;
        end
    end
    fclose(fid);

    % --- 3. 解析关键参数 ---
    % 检查 lines, samples, bands 是否存在
    if isfield(info, 'lines'), lines = str2double(info.lines); else, error('HDR缺少 lines 字段'); end
    if isfield(info, 'samples'), samples = str2double(info.samples); else, error('HDR缺少 samples 字段'); end
    if isfield(info, 'bands'), bands = str2double(info.bands); else, error('HDR缺少 bands 字段'); end
    
    header_offset = 0;
    if isfield(info, 'header_offset'), header_offset = str2double(info.header_offset); end
    
    interleave = 'bsq';
    if isfield(info, 'interleave'), interleave = lower(info.interleave); end
    
    % 处理字节序 (Byte Order)
    machine = 'ieee-le'; % 默认为小端 (Intel/Windows)
    if isfield(info, 'byte_order')
        if str2double(info.byte_order) == 1
            machine = 'ieee-be'; % 大端 (Unix/Network)
        end
    end

    % --- 4. 处理数据类型 (修复点) ---
    dtype_map = containers.Map({1, 2, 3, 4, 5, 12}, ...
        {'uint8', 'int16', 'int32', 'single', 'double', 'uint16'});
    
    % 兼容不同的写法: data_type 或 datatype
    if isfield(info, 'data_type')
        dt_code = str2double(info.data_type);
    elseif isfield(info, 'datatype')
        dt_code = str2double(info.datatype);
    else
        % 调试信息：如果报错，打印出所有读取到的字段名
        disp('读取到的字段名有:');
        disp(fieldnames(info));
        error('HDR文件中缺少 data_type 字段 (解析失败)');
    end

    if isKey(dtype_map, dt_code)
        precision = dtype_map(dt_code);
    else
        error(['不支持的数据类型代码: ' num2str(dt_code)]); 
    end

    % --- 5. 读取二进制数据 ---
    data = multibandread(datfile, [lines, samples, bands], ...
                         precision, header_offset, interleave, machine);
end