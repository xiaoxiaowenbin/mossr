function rep = GridIndex(rep, nGrid)
Grid = CreateGrid(rep, nGrid); 
    for i = 1:numel(rep)
        rep(i) = FindGridIndex(rep(i), Grid);
    end
end

%% func_CreateGrid
function Grid = CreateGrid(pop, nGrid)
    alpha = 0.1;
    
    % 强制提取 Cost 并转置，确保每列是一个个体的 Cost
    % 假设每个个体的 Cost 是 [f1; f2] 或 [f1, f2]
    all_costs = [pop.Cost]; 
    
    % 如果 all_costs 是 1x(nPop*nObj)，需要 reshape
    % 最稳妥的办法是：
    nPop = numel(pop);
    nObj = numel(pop(1).Cost);
    c = reshape(all_costs, nObj, nPop); % 确保 c 是 [nObj x nPop]
   
    cmin = min(c, [], 2); 
    cmax = max(c, [], 2); 
    
    dc = cmax - cmin;          
    cmin = cmin - alpha * dc;
    cmax = cmax + alpha * dc;
    
    empty_grid.LB = [];
    empty_grid.UB = [];
    Grid = repmat(empty_grid, nObj, 1);
    
    for j = 1:nObj
        % 生成刻度
        cj = linspace(cmin(j), cmax(j), nGrid + 1);
        % 加上无穷边界确保覆盖所有点
        Grid(j).LB = [-inf, cj];
        Grid(j).UB = [cj, +inf];
    end
end

%% func_FindGridIndex
function particle = FindGridIndex(particle, Grid)

    nObj = numel(particle.Cost);
    
    nGrid = numel(Grid(1).LB);
    
    particle.GridSubIndex = zeros(1, nObj); 
    
    for j = 1:nObj
        
        particle.GridSubIndex(j) = ...
            find(particle.Cost(j)<Grid(j).UB, 1, 'first');
    end

    particle.GridIndex = particle.GridSubIndex(1);
    for j = 2:nObj
        particle.GridIndex = particle.GridIndex-1;
        particle.GridIndex = nGrid*particle.GridIndex;
        particle.GridIndex = particle.GridIndex+particle.GridSubIndex(j);
    end
    
end