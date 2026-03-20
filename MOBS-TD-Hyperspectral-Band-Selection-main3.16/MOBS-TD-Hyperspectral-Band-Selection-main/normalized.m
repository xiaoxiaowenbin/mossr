function [value] = normalized(map)
% normalized
map = double(map);
maxx=max(map(:));
minn=min(map(:));
%%首先，map - minn：将输入矩阵中的每个元素减去最小值minn，这样可以将最小值平移到0。
%%然后，(map - minn) / (maxx - minn)：将上述结果除以最大值和最小值之间的差值(maxx - minn)，这一步将数据缩放到[0, 1]的范围内
value=(map-minn)/(maxx-minn);     
end