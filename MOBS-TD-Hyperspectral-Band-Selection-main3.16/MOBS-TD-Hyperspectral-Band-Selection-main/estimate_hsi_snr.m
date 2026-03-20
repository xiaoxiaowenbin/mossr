function snr_vec = estimate_hsi_snr(data_cube)
    % data_cube: 输入的高光谱数据，维度为 [Height, Width, Bands]
    [H, W, L] = size(data_cube);
    snr_vec = zeros(L, 1);
    
    for i = 1:L
        band = double(data_cube(:,:,i));
        
        % 1. 计算局部均值 (信号)
        local_avg = mean(band(:)); 
        
        % 2. 估算噪声 (使用局部差异或全局标准差的改进版)
        % 简单做法：利用高通滤波移除空间相关性，剩下的视为噪声
        % 这里使用一种常用的空间差分法估算噪声标准差
        diff_h = diff(band, 1, 1); % 水平差分
        noise_std = std(diff_h(:)) / sqrt(2); % 差分后的标准差需修正
        
        % 3. 计算 SNR (分贝 dB)
        if noise_std > 0
            snr_vec(i) = 20 * log10(local_avg / noise_std);
        else
            snr_vec(i) = 0;
        end
    end
end