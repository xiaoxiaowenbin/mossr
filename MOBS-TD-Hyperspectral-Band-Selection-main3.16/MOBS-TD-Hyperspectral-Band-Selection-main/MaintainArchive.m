function rep = MaintainArchive(rep, nRep, nGrid)
    rep = DetermineDomination(rep);
    rep = rep(~[rep.IsDominated]);
    rep = GridIndex(rep, nGrid);
    
    if numel(rep) > nRep && numel(rep) > 5
        Extra = numel(rep) - nRep;
        try
            % 获取排序：seq(1) 是得分最高的，seq(end) 是得分最低的
            seq = WSIS(rep);
            % 精确删除得分最低的解
            disp('精确删除得分最低的解');
            rep(seq(end-Extra+1:end)) = []; 
        catch
            % 兜底逻辑
            rep(randperm(numel(rep), Extra)) = [];
        end
    end
end