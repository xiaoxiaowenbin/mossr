%============================= 
% Defining Pareto dominance
%============================= 
function pop = DetermineDomination(pop)
    nPop = numel(pop);   
    %初始化均为否
    for i = 1:nPop
        pop(i).IsDominated = false;
    end
    
    for i = 1:nPop-1
        for j = i+1:nPop
            %%对比两个备选解是否存在帕雷特支配关系
            if Dominates(pop(i), pop(j))
               pop(j).IsDominated = true;  
            end
            
            if Dominates(pop(j), pop(i))
               pop(i).IsDominated = true;
            end
            
            if pop(i).Position == pop(j).Position
                pop(j).IsDominated = true;
            end
        end
    end

end

