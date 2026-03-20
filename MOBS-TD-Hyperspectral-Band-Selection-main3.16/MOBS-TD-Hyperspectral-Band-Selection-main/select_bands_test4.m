function [selected_bands, info] = select_bands_test4(x_pareto, f_pareto, num_bands, K, snr_vec, target_d)
% Reliability-Calibrated Hybrid Suppression for Band Selection (RCHS-BS)
%
% 论文主方法版：
%   1) 多精英解融合得到 band importance
%   2) 基于 SNR / 激活稳定性 / 跨精英一致性 建模可靠性
%   3) 对极前端施加保守小硬切
%   4) 判别是否存在“显著低可靠前缀”
%   5) 若显著前缀成立，则启动条件硬切
%   6) 对前段过渡区施加强软抑制
%   7) 对全谱施加温和软抑制
%   8) 候选池约束 + 邻域抑制贪婪选 K 个 band
%
% 输入:
%   x_pareto   : [N x L^2] Pareto archive，每行是一组展开后的 W
%   f_pareto   : [N x M]   对应多目标值
%   num_bands  : L         总波段数
%   K          :           最终选取波段数
%   snr_vec    : [L x 1]   band-wise SNR
%   target_d   : [L x 1]   target prior spectrum
%
% 输出:
%   selected_bands : [K x 1] 最终波段索引
%   info           : 调试 / 作图 / 论文分析信息
%
% 说明:
%   - 本实现不是“固定大硬切”版本；
%   - 它是“小硬切先验 + 显著低可靠前缀判别 + 条件硬切 + 前段强化软抑制”的主方法版；
%   - 若数据集不存在明显低可靠前缀，则只会保留小硬切和软抑制，不会激进切掉整段前缀。

    %% ==================== 0) 输入检查 ====================
    target_d = target_d(:);
    snr_vec  = snr_vec(:);

    if size(x_pareto, 1) ~= size(f_pareto, 1)
        error('x_pareto 和 f_pareto 的解数必须一致。');
    end

    if length(snr_vec) ~= num_bands || length(target_d) ~= num_bands
        error('snr_vec 和 target_d 的长度必须等于 num_bands。');
    end

    num_solutions = size(x_pareto, 1);
    if num_solutions == 0
        error('Pareto archive 为空。');
    end

    if K <= 0 || K > num_bands
        error('K 必须满足 1 <= K <= num_bands。');
    end

    %% ==================== 1) 参数 ====================
    p = get_default_params(num_bands, K);

    %% ==================== 2) 精英解选择 ====================
    f_min  = min(f_pareto, [], 1);
    f_max  = max(f_pareto, [], 1);
    f_norm = (f_pareto - f_min) ./ (f_max - f_min + eps);

    dist_to_ideal = sqrt(sum(f_norm.^2, 2));
    [~, sort_idx] = sort(dist_to_ideal, 'ascend');

    num_elites    = min(p.num_elites, num_solutions);
    elite_indices = sort_idx(1:num_elites);

    elite_dist = dist_to_ideal(elite_indices);
    tau = mean(elite_dist) + eps;
    elite_w = exp(-elite_dist / tau);
    elite_w = elite_w / sum(elite_w);

    %% ==================== 3) 多精英融合 importance ====================
    elite_energy_mat = zeros(num_bands, num_elites);
    activation_count = zeros(num_bands, 1);

    for i = 1:num_elites
        idx = elite_indices(i);

        W_temp = reshape(x_pareto(idx, :), num_bands, num_bands);
        w_energy_single = sum(W_temp.^2, 2);

        elite_energy_mat(:, i) = w_energy_single;

        th = mean(w_energy_single) + p.active_std_factor * std(w_energy_single);
        active_mask = (w_energy_single >= th);
        activation_count(active_mask) = activation_count(active_mask) + 1;
    end

    fused_energy  = elite_energy_mat * elite_w(:);
    w_energy_norm = normalize01(fused_energy);

    %% ==================== 4) 稳定性 / 一致性 / target prior ====================
    stability_score = activation_count / num_elites;

    elite_energy_norm = normalize_columns01(elite_energy_mat);
    elite_dispersion  = std(elite_energy_norm, 0, 2);
    consistency_score = 1 - normalize01(elite_dispersion);

    target_prior_norm = normalize01(abs(target_d));

    %% ==================== 5) 基础分数 ====================
    % 保留你当前高表现骨架中的“乘法主干”
    base_score = w_energy_norm .* target_prior_norm;
    base_score = base_score .* (p.stability_base + p.stability_gain * stability_score);

    %% ==================== 6) 基础无效 mask ====================
    snr_threshold = median(snr_vec) * p.snr_factor;
    noise_bands = find(snr_vec < snr_threshold);

    edge_bands = unique([(1:p.edge_width)'; ((num_bands-p.edge_width+1):num_bands)']);
    invalid_bands = unique([noise_bands(:); edge_bands(:)]);

    %% ==================== 7) 可靠性建模 ====================
    % reliability 只反映“质量/可信度”，不直接混入 target prior
    snr_norm = normalize01(snr_vec);

    reliability = ...
        p.rel_w_snr  * snr_norm + ...
        p.rel_w_stab * stability_score + ...
        p.rel_w_cons * consistency_score;

    reliability   = normalize01(reliability);
    reliability_s = movmean(reliability, p.smooth_win, 'Endpoints', 'shrink');
    reliability_s = normalize01(reliability_s);

    %% ==================== 8) 小硬切先验 ====================
    % 这一步始终保守生效，只处理极前端
    hard_prefix_prior = min(p.hard_prefix_len, num_bands);
    if hard_prefix_prior > 0
        invalid_bands = unique([invalid_bands; (1:hard_prefix_prior)']);
    end

    %% ==================== 9) 显著低可靠前缀判别 ====================
    % 在小硬切之后，只对前段搜索区做“是否存在显著低可靠前缀”的判别
    prefix_search_end = min(max(hard_prefix_prior + 1, p.prefix_search_end), num_bands);

    front_zone = [];
    if prefix_search_end > hard_prefix_prior
        front_zone = (hard_prefix_prior + 1):prefix_search_end;
    end

    body_l = min(num_bands, max(prefix_search_end + 1, round(p.body_start_ratio * num_bands)));
    body_r = min(num_bands, max(body_l, round(p.body_end_ratio   * num_bands)));
    body_zone = body_l:body_r;

    if isempty(front_zone)
        front_ref = NaN;
        body_ref  = median(reliability_s);
        front_gap = 0;
    else
        front_ref = median(reliability_s(front_zone));
        if isempty(body_zone)
            body_ref = median(reliability_s);
        else
            body_ref = median(reliability_s(body_zone));
        end
        front_gap = max(body_ref - front_ref, 0);
    end

    % 恢复阈值：从前段低可靠基线向主体区插值
    recover_thr = front_ref + p.recover_ratio * front_gap;
    if isnan(recover_thr)
        recover_thr = median(reliability_s);
    end

    recover_idx = [];
    if ~isempty(front_zone)
        search_l = hard_prefix_prior + 1;
        search_r = prefix_search_end - p.consec_len + 1;

        if search_r >= search_l
            for i = search_l:search_r
                seg = reliability_s(i:(i + p.consec_len - 1));
                if all(seg >= recover_thr)
                    recover_idx = i;
                    break;
                end
            end
        end
    end

    % 是否显著低可靠前缀：
    % 1) 前段和主体区差异够明显
    % 2) 确实存在持续恢复点
    enable_hardcut = (~isempty(recover_idx)) && (front_gap >= p.min_front_gap);

    if enable_hardcut
        B_cut = recover_idx - 1;
        B_cut = max(hard_prefix_prior, B_cut);
    else
        B_cut = hard_prefix_prior;
    end

    B_cut = max(0, min(B_cut, prefix_search_end));

    % 条件硬切部分（不重复切小硬切先验）
    if enable_hardcut && B_cut > hard_prefix_prior
        extra_hardcut_bands = ((hard_prefix_prior + 1):B_cut)';
        invalid_bands = unique([invalid_bands; extra_hardcut_bands]);
    else
        extra_hardcut_bands = [];
    end

    %% ==================== 10) 全局软抑制 ====================
    global_gate = p.global_soft_floor + ...
                  (1 - p.global_soft_floor) * (reliability_s .^ p.global_soft_gamma);

    %% ==================== 11) 前段强化软抑制 ====================
    transition_end = min(max(hard_prefix_prior + 1, p.transition_end), num_bands);

    transition_zone = [];
    if transition_end > hard_prefix_prior
        transition_zone = (hard_prefix_prior + 1):transition_end;
    end

    if isempty(transition_zone)
        transition_penalty_strength = 0;
        transition_base_gate_full = nan(num_bands, 1);
        transition_gate = ones(num_bands, 1);
    else
        % 前段 gap 越大，说明前段越像低可靠过渡区
        transition_penalty_strength = min(1, front_gap / p.transition_gap_scale);

        transition_base_gate = p.transition_soft_floor + ...
            (1 - p.transition_soft_floor) * ...
            (reliability_s(transition_zone) .^ p.transition_soft_gamma);

        transition_scale = 1 - transition_penalty_strength * (1 - p.transition_scale_floor);

        transition_gate_zone = transition_base_gate * transition_scale;
        transition_gate_zone = max(p.transition_gate_min, transition_gate_zone);

        transition_gate = ones(num_bands, 1);
        transition_gate(transition_zone) = transition_gate_zone;

        transition_base_gate_full = nan(num_bands, 1);
        transition_base_gate_full(transition_zone) = transition_base_gate;
    end

    %% ==================== 12) 最终 score ====================
    reliability_gate = global_gate .* transition_gate;

    band_scores = base_score .* reliability_gate;
    band_scores(invalid_bands) = -inf;

    %% ==================== 13) 候选池限制 ====================
    candidate_pool = [];
    exclusion_window = max(p.min_exclusion_window, floor(num_bands / (K * p.exclusion_div)));

    valid_idx = find(isfinite(band_scores));

    if isempty(valid_idx)
        warning('所有波段均被抑制/屏蔽，退化为按 SNR 选取。');

        fallback_pool = setdiff((1:num_bands)', invalid_bands);
        if isempty(fallback_pool)
            fallback_pool = setdiff((1:num_bands)', edge_bands);
        end
        if isempty(fallback_pool)
            fallback_pool = (1:num_bands)';
        end

        [~, order_tmp] = sort(snr_vec(fallback_pool), 'descend');
        selected_bands = fallback_pool(order_tmp(1:min(K, length(fallback_pool))));
        selected_bands = sort(selected_bands);

        info = build_info_struct();
        return;
    end

    [~, score_order] = sort(band_scores(valid_idx), 'descend');
    pool_size = min(max(p.pool_mult * K, p.pool_min), length(valid_idx));

    candidate_pool = valid_idx(score_order(1:pool_size));

    candidate_mask = false(num_bands, 1);
    candidate_mask(candidate_pool) = true;

    temp_scores = band_scores;
    temp_scores(~candidate_mask) = -inf;

    %% ==================== 14) 贪婪选择 + 邻域抑制 ====================
    selected_bands = [];

    for k = 1:K
        [max_val, current_best_idx] = max(temp_scores);
        if ~isfinite(max_val)
            break;
        end

        selected_bands = [selected_bands; current_best_idx]; %#ok<AGROW>

        low_bound = max(1, current_best_idx - exclusion_window);
        up_bound  = min(num_bands, current_best_idx + exclusion_window);
        temp_scores(low_bound:up_bound) = -inf;
    end

    %% ==================== 15) 兜底补足 ====================
    selected_bands = unique(selected_bands(:), 'stable');

    if length(selected_bands) < K
        valid_pool = setdiff((1:num_bands)', [selected_bands; invalid_bands(:)]);
        if ~isempty(valid_pool)
            [~, idx_sort] = sort(band_scores(valid_pool), 'descend');
            n_need = K - length(selected_bands);
            fill_bands = valid_pool(idx_sort(1:min(n_need, length(valid_pool))));
            selected_bands = [selected_bands; fill_bands(:)];
        end
    end

    selected_bands = sort(selected_bands);
    if length(selected_bands) > K
        selected_bands = selected_bands(1:K);
    end

    %% ==================== 16) 输出信息 ====================
    info = build_info_struct();

    % ==================== nested info builder ====================
    function s = build_info_struct()
        s = struct();

        s.params = p;

        s.elite_indices = elite_indices;
        s.elite_weights = elite_w;
        s.dist_to_ideal = dist_to_ideal;

        s.fused_energy = fused_energy;
        s.w_energy_norm = w_energy_norm;

        s.stability_score = stability_score;
        s.consistency_score = consistency_score;
        s.target_prior_norm = target_prior_norm;
        s.snr_norm = snr_norm;

        s.reliability = reliability;
        s.reliability_s = reliability_s;

        s.base_score = base_score;
        s.global_gate = global_gate;
        s.transition_gate = transition_gate;
        s.transition_base_gate = transition_base_gate_full;
        s.reliability_gate = reliability_gate;
        s.band_scores = band_scores;

        s.hard_prefix_prior = hard_prefix_prior;
        s.prefix_search_end = prefix_search_end;
        s.transition_end = transition_end;

        s.front_zone = front_zone;
        s.body_zone = body_zone;
        s.front_ref = front_ref;
        s.body_ref = body_ref;
        s.front_gap = front_gap;

        s.recover_thr = recover_thr;
        s.recover_idx = recover_idx;
        s.enable_hardcut = enable_hardcut;
        s.B_cut = B_cut;
        s.extra_hardcut_bands = extra_hardcut_bands;

        s.transition_penalty_strength = transition_penalty_strength;

        s.snr_threshold = snr_threshold;
        s.noise_bands = noise_bands;
        s.edge_bands = edge_bands;
        s.invalid_bands = invalid_bands;

        s.candidate_pool = candidate_pool;
        s.exclusion_window = exclusion_window;
        s.selected_bands = selected_bands;
    end
end

%% ======================== local helpers ========================

function p = get_default_params(num_bands, K)
    p = struct();

    % ----- elite fusion -----
    p.num_elites = 8;
    p.active_std_factor = 0.35;

    % ----- base score -----
    p.stability_base = 0.92;
    p.stability_gain = 0.08;

    % ----- reliability weights -----
    p.rel_w_snr  = 0.45;
    p.rel_w_stab = 0.30;
    p.rel_w_cons = 0.25;

    % ----- basic invalid mask -----
    p.snr_factor = 0.60;
    p.edge_width = max(3, round(0.02 * num_bands));

    % ----- small hard prefix prior -----
    % 对 120~200 波段数据通常会落在 6~8 左右
    p.hard_prefix_len = min(8, max(3, round(0.06 * num_bands)));

    % ----- prefix search / transition zone -----
    % 搜索显著低可靠前缀时，只在前段搜索区判断
    p.prefix_search_end = min(25, max(p.hard_prefix_len + 4, round(0.20 * num_bands)));

    % 前段强化软抑制区
    p.transition_end = min(25, max(p.hard_prefix_len + 4, round(0.20 * num_bands)));

    % ----- body region for comparison -----
    p.body_start_ratio = 0.35;
    p.body_end_ratio   = 0.75;

    % ----- prefix detection -----
    p.recover_ratio = 0.80;          % 恢复阈值从前段到主体区插值比例
    p.consec_len    = max(3, round(0.03 * num_bands));
    p.min_front_gap = 0.10;          % 显著低可靠前缀触发最小 gap

    % ----- global soft suppression -----
    p.global_soft_floor = 0.78;
    p.global_soft_gamma = 0.90;

    % ----- stronger front suppression -----前端软抑制主要参数
    p.transition_soft_floor = 0.22;
    p.transition_soft_gamma = 2.00;

    % 当前段显著低于主体区时，再额外统一下压
    p.transition_gap_scale   = 0.10;
    p.transition_scale_floor = 0.40;
    p.transition_gate_min    = 0.08;

    % ----- reliability smoothing -----
    sw = max(5, round(0.03 * num_bands));
    if mod(sw, 2) == 0
        sw = sw + 1;
    end
    p.smooth_win = sw;

    % ----- candidate pool + redundancy suppression -----
    p.pool_mult = 4;
    p.pool_min  = max(40, round(1.5 * K));

    p.exclusion_div = 4;
    p.min_exclusion_window = max(2, round(0.015 * num_bands));
end

function y = normalize01(x)
    xmin = min(x);
    xmax = max(x);
    y = (x - xmin) ./ (xmax - xmin + eps);
end

function Xn = normalize_columns01(X)
    Xn = zeros(size(X));
    for c = 1:size(X, 2)
        Xn(:, c) = normalize01(X(:, c));
    end
end