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