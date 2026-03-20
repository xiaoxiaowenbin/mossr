clc;clear;close all;
%{ 
    Reference: X. Sun, P. Lin, X. Shang, H. Pang and X. Fu, "MOBS-TD: Multiobjective 
               Band Selection With Ideal Solution Optimization Strategy for Hyperspectral 
               Target Detection,” IEEE Journal of Selected Topics in Applied Earth 
               Observations and Remote Sensing (IEEE JSTARS), DOI: 10.1109/JSTARS.2024.3402381.
%}

%% General Setup
nRuns = 10;       % number of independent runs (Total execution times)
seed_base = 123;             % 种子 123-132

%% Parameters (Algorithm specific)
MaxT = 200;        % Maximum number of iterations
nPop = 150;        % population size
nRep = 20;         % Candidate solution set size

% PSO Parameters
w_start = 0.5;     % Initial inertia factor (variable inside loop)
wdamp = 0.99;      % Attenuation factor of inertia
c1 = 1;            % Global learning factor
c2 = 1;            % Individual learning factor

nGrid = 4;         % Number of grids per dimension (nGrid+1)
mu = 0.1;          % mutation probability regulator
maxrate = 0.2;     % Evolutionary speed control factor

%% Load Data & Pre-processing (Perform ONCE to save time)
disp('Loading and Pre-processing data...');
load abu-urban-5.mat;
img_src = data;
img_gt = map;

[W, H, L] = size(img_src);
img_src = normalized(img_src);
img = reshape(img_src, W * H, L);
d = get_target(img, img_gt);

% These calculations are heavy and deterministic, calculate once.
En = Entrop(img); 
D = spectral_spatial(img);

%% ========== VD估计（确定K值）==========
% 这一步决定选多少个波段
t_vd = 1e-4; 
fprintf('执行 VD 估计 (HFC/NWHFC)...\n');
hfc_vd = HFC1(img_src, t_vd);
nwhfc_vd = NWHFC(img_src, t_vd);
nVar = 2 * max(hfc_vd, nwhfc_vd); % 通常取 HFC 和 NWHFC 较大值的两倍作为冗余
fprintf('最终选择的波段数量 (K): %d\n', nVar);

%% Cost Function Definition
% @(x,h,d,m,t) 表示这是一个匿名函数
CostFunction = @(x,h,d,m,t) BS_model(x,h,d,m,t);

%% Initialization of Storage for 10 Runs
% Structure to store results of each run
RunResults = repmat(struct('Solution', [], 'AUC_PFPD', 0, 'AUC_tauPD', 0, 'AUC_tauPF', 0), nRuns, 1);

%% Main Loop for Independent Runs
for run = 1:nRuns
    fprintf('-------------------------------------------\n');
    fprintf('Starting Run %d / %d \n', run, nRuns);
    
    current_seed = seed_base + run - 1;
    rng(current_seed, 'twister');
    fprintf('本轮随机种子: %d\n', current_seed);
    
    %% Initialization for Current Run
    VarSize = [1 nVar];                            
    VarMin = 1;                                    
    VarMax = L;                                    
    
    w = w_start; % Reset inertia for each run

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

    % 1st-generation population initialization
    for i = 1:nPop 
        pop(i).Position = sort(randperm(VarMax, nVar));
        pop(i).Velocity = zeros(VarSize);
        pop(i).Cost = CostFunction(pop(i).Position, En, D, img', d);
        pop(i).Best.Position = pop(i).Position;
        pop(i).Best.Cost = pop(i).Cost;
    end

    % Determine Domination
    pop = DetermineDomination(pop);
    rep = pop(~[pop.IsDominated]);

    % Grid
    rep = GridIndex(rep, nGrid);

    %% Optimization Loop (MaxT)
    for it = 1:MaxT
        for i = 1:nPop
            % Select leader
            leader = SelectLeader(rep);
            
            % Update Velocity
            pop(i).Velocity = w*pop(i).Velocity ...
                +c1*rand(VarSize).*(pop(i).Best.Position-pop(i).Position) ...
                +c2*rand(VarSize).*(leader.Position-pop(i).Position);
            
            % Limitation
            pop(i).Velocity = max(pop(i).Velocity, (-1)*maxrate*VarMax);
            pop(i).Velocity = min(pop(i).Velocity, maxrate*VarMax);
            pop(i).Velocity = fix(pop(i).Velocity);
            
            % Update Position
            pop(i).Position = pop(i).Position + pop(i).Velocity;
            
            % Regularization
            pop(i).Position = limitPositionVariables(pop(i).Position, VarMin, VarMax);
            
            % Update Fitness
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
        
        %% Crossover in Rep Set
        pc = (1-(it-1)/(MaxT-1))^(1/mu);
        num_rep = numel(rep);
        if rand < pc
            nCrossover = 2*floor(pc*num_rep/2);
            popc = repmat(empty_particle, nCrossover/2, 1); % childs
            cross_index = reshape(randperm(num_rep,nCrossover),nCrossover/2,2);
            for k = 1:nCrossover/2
                p1 = rep(cross_index(k,1));
                p2 = rep(cross_index(k,2));
                
                popc(k).Position = Crossover(p1.Position, p2.Position, En); % Crossover
                
                % Velocity update for children
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
        
        % Update Grid
        rep = GridIndex(rep, nGrid);

        % Check if rep set is full
        if numel(rep) > nRep
            Extra = numel(rep) - nRep;
            seq = WSIS(rep);
            for e = 1:Extra
                rep = DeleteRepMemebr(rep, seq);
            end        
        end

        % Plot Costs (Only update occasionally or for the first run to save time/GUI lag)
        if mod(it, 20) == 0 || it == 1
            figure(1);
            PlotCosts(pop, rep);
            title(['Run: ', num2str(run), ' | Iteration: ', num2str(it), ' | Rep Size: ', num2str(numel(rep))]);
            pause(0.01);
        end
        
        fprintf('rep数：%d\n',numel(rep))
        % Damping Inertia Weight
        w = w * wdamp;
    end
    
    %% Run Evaluation (MSR + AUC)
    disp(['Run ' num2str(run) ' Optimization Finished. Calculating AUC...']);
    
    repSet = {rep.Position};
    detector_Name = 'CEM';
    
    % Select the single best solution from the Pareto Front of this run
    fSolution = MSR(repSet, detector_Name, img, W, H, d);
    
    % Perform Detection
    detectmap = reshape(detector(img(:,fSolution), d(fSolution)', detector_Name), W, H);
    
    % Calculate AUC
    det_map_vec = detectmap(:);
    GT_vec = img_gt(:);
    [AUC, ~] = cal_AUC(det_map_vec, GT_vec, 1, 1);
    
    % Store Results
    RunResults(run).Solution = fSolution;
    RunResults(run).AUC_PFPD = AUC.PFPD;
    RunResults(run).AUC_tauPD = AUC.tauPD;
    RunResults(run).AUC_tauPF = AUC.tauPF;
    
    disp(['Run ' num2str(run) ' Result -> AUC(PF-PD): ' num2str(AUC.PFPD)]);
end

%% Statistical Analysis & Final Output
disp('===========================================================');
disp('                  FINAL STATISTICS (10 Runs)               ');
disp('===========================================================');

% Extract AUCs to a vector
all_AUCs = [RunResults.AUC_PFPD];

% Calculate Mean and Std
mean_AUC = mean(all_AUCs);
std_AUC = std(all_AUCs);
best_AUC = max(all_AUCs);
worst_AUC = min(all_AUCs);
[~, best_run_idx] = max(all_AUCs);

fprintf('Mean AUC (PF-PD): %.6f\n', mean_AUC);
fprintf('Std  AUC (PF-PD): %.6f\n', std_AUC);
fprintf('Best AUC (PF-PD): %.6f (Run %d)\n', best_AUC, best_run_idx);
fprintf('Worst AUC(PF-PD): %.6f\n', worst_AUC);

%% Visualization of the Best Run
disp(['Plotting detection map for the Best Run (Run ' num2str(best_run_idx) ')...']);

best_Solution = RunResults(best_run_idx).Solution;
detectmap_best = reshape(detector(img(:,best_Solution), d(best_Solution)', detector_Name), W, H);

figure('Name', ['Best Result (Run ' num2str(best_run_idx) ')']);
imagesc(detectmap_best);
colorbar;
title(['Best Detection Map (AUC: ' num2str(best_AUC) ')']);
axis image;

disp(['Optimal band subset (Best Run): (' num2str(best_Solution) ')']);