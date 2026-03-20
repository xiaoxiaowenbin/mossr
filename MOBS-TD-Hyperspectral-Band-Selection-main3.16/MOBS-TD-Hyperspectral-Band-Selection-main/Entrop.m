function H = Entrop(X)
    %%矩阵X进行转置
    X = X';
    G = 256;
    [L, N] = size(X);
    %%初始化一个列向量H，大小为L x 1，用来存储每个特征的熵值
    H = zeros(L, 1);
    minX = min(X(:)); 
    maxX = max(X(:));
    edge = linspace(minX, maxX, G);
    for i = 1 : L
        histX = hist(X(i, :), edge) / N;
        %%信息熵的计算
        H (i) = - histX * log(histX + eps)';
    end  
end
