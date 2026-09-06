function main_radar_synthetic()
% This program evaluates NSCG-M and NSCG-S on the
%  geometry :8 transmitters, 8 receivers, all 64 bistatic pairs, x_star=[400;100] m). It studies symmetric
%  alpha-stable measurement noise over alpha in {1.2, 1.5, 1.8},  dispersion in {10,20,30,40,50} m.
%
%   synthetic_rmse_vs_dispersion_multi_alpha.fig/png/pdf
%   synthetic_geometry_paths_alpha_1p5_gamma_30.fig/png/pdf
%   synthetic_final_scatter_alpha_1p5_gamma_30.fig/png/pdf
%   synthetic_results_multi_alpha.mat

    clc;
    close all;

    cfg = default_config();
    validate_config(cfg);
    problem = build_literature_geometry(cfg);

    project_root = fileparts(mfilename('fullpath'));
    output_dir = fullfile(project_root, cfg.output_folder);
    if exist(output_dir, 'dir') ~= 7
        mkdir(output_dir);
    end

    fprintf('============================================================\n');
    fprintf('Robust elliptic localization under S-alpha-S noise\n');
    fprintf('N_t=%d, N_r=%d, pairs=%d, target=[%.1f, %.1f] m\n', ...
        problem.Nt, problem.Nr, problem.number_of_pairs, ...
        problem.x_star_m(1), problem.x_star_m(2));
    fprintf('alpha values: ');
    fprintf('%.1f ', cfg.alpha_values);
    fprintf('\ndispersion values (m): ');
    fprintf('%.1f ', cfg.dispersion_values_m);
    fprintf('\nrepeated measurements per pair=%d\n', ...
        cfg.repeats_per_pair);
    fprintf('Monte Carlo runs=%d, budget per curve=%.0f evaluations\n', ...
        cfg.num_runs, cfg.sample_budget);
    fprintf('============================================================\n');

    number_of_cases = numel(cfg.alpha_values) * ...
        numel(cfg.dispersion_values_m);
    case_results = repmat(empty_case_result(), 1, number_of_cases);
    case_index = 0;

    for alpha_index = 1:numel(cfg.alpha_values)
        alpha = cfg.alpha_values(alpha_index);

        for dispersion_index = 1:numel(cfg.dispersion_values_m)
            dispersion_m = cfg.dispersion_values_m(dispersion_index);
            case_index = case_index + 1;

            checkpoints = make_record_checkpoints( ...
                cfg.sample_budget, cfg.num_record_points);
            raw = initialize_raw(cfg, problem.d, numel(checkpoints));

            fprintf('\n------------------------------------------------------------\n');
            fprintf('Case %d/%d: alpha=%.1f, dispersion=%.1f m\n', ...
                case_index, number_of_cases, alpha, dispersion_m);
            fprintf('------------------------------------------------------------\n');

            for run = 1:cfg.num_runs
                data_seed = make_data_seed(cfg, alpha, dispersion_m, run);
                common_algorithm_seed = make_algorithm_seed( ...
                    cfg, alpha, dispersion_m, run);
                frame = generate_measurement_frame( ...
                    problem, cfg, alpha, dispersion_m, data_seed);

                fprintf('Run %3d/%3d, data seed=%d, algorithm seed=%d\n', ...
                    run, cfg.num_runs, data_seed, common_algorithm_seed);

                for method_index = 1:numel(cfg.method_order)
                    method_name = cfg.method_order{method_index};
                    start_time = tic;

                    if strcmp(method_name, 'NSCG_M')
                        trajectory = run_nscg_m(problem, frame, ...
                            cfg.alg.NSCG_M, cfg, checkpoints, ...
                            common_algorithm_seed);
                    else
                        trajectory = run_nscg_s(problem, frame, ...
                            cfg.alg.NSCG_S, cfg, checkpoints, ...
                            common_algorithm_seed);
                    end

                    elapsed_time = toc(start_time);
                    raw.(method_name).location_error_m(run, :) = ...
                        trajectory.location_error_m;
                    raw.(method_name).br_rmse_m(run, :) = ...
                        trajectory.br_rmse_m;
                    raw.(method_name).stationarity(run, :) = ...
                        trajectory.stationarity;
                    raw.(method_name).position_m(run, :, :) = reshape( ...
                        trajectory.position_m, ...
                        [1, problem.d, numel(checkpoints)]);
                    raw.(method_name).runtime_s(run) = elapsed_time;

                    fprintf(['  %-7s final error=%9.3e m, ', ...
                        'stationarity=%9.3e, time=%7.2fs\n'], ...
                        cfg.method_labels.(method_name), ...
                        trajectory.location_error_m(end), ...
                        trajectory.stationarity(end), elapsed_time);
                end
            end

            summary = aggregate_case(raw, checkpoints, cfg, problem);
            statistics = final_case_statistics( ...
                raw, cfg, problem, alpha, dispersion_m);

            case_results(case_index).alpha = alpha;
            case_results(case_index).dispersion_m = dispersion_m;
            case_results(case_index).checkpoints = checkpoints;
            case_results(case_index).raw = raw;
            case_results(case_index).summary = summary;
            case_results(case_index).statistics = statistics;

            fprintf('Final statistics, alpha=%.1f, dispersion=%.1f m:\n', ...
                alpha, dispersion_m);
            for method_index = 1:numel(cfg.method_order)
                method_name = cfg.method_order{method_index};
                stats = statistics.(method_name);
                fprintf(['  %-7s RMSE=%8.3f m [%.3f, %.3f], ', ...
                    'median=%8.3f m, q90=%8.3f m\n'], ...
                    cfg.method_labels.(method_name), stats.rmse_m, ...
                    stats.rmse_ci95_m(1), stats.rmse_ci95_m(2), ...
                    stats.median_m, stats.q90_m);
            end

            if cfg.save_partial_mat
                completed_case_results = case_results(1:case_index); %#ok<NASGU>
                save(fullfile(output_dir, ...
                    'synthetic_results_multi_alpha_partial.mat'), ...
                    'completed_case_results', 'cfg', 'problem', '-v7.3');
            end
        end
    end

    if cfg.save_mat
        save(fullfile(output_dir, ...
            'synthetic_results_multi_alpha.mat'), ...
            'case_results', 'cfg', 'problem', '-v7.3');
    end

    fig = plot_rmse_vs_dispersion(case_results, cfg);
    save_figure_files(fig, fullfile(output_dir, ...
        'synthetic_rmse_vs_dispersion_multi_alpha'), cfg);
    close_if_requested(fig, cfg);

    reference_case = find_reference_case(case_results, ...
        cfg.reference_alpha, cfg.reference_dispersion_m);

    fig = plot_geometry_and_paths(reference_case, problem, cfg);
    save_figure_files(fig, fullfile(output_dir, ...
        make_reference_name_2d('synthetic_geometry_paths', ...
        reference_case.alpha, reference_case.dispersion_m)), cfg);
    close_if_requested(fig, cfg);

    fig = plot_final_scatter(reference_case, problem, cfg);
    save_figure_files(fig, fullfile(output_dir, ...
        make_reference_name_2d('synthetic_final_scatter', ...
        reference_case.alpha, reference_case.dispersion_m)), cfg);
    close_if_requested(fig, cfg);

    fprintf('\nAll synthetic results were written to:\n  %s\n', output_dir);
end

%% ========================================================================
% Configuration
% =========================================================================
function cfg = default_config()
    cfg.problem_seed = 20260904;
    cfg.data_seed = 41021;
    cfg.base_seed = 73129;
    cfg.bootstrap_seed = 918273;

    cfg.num_runs = 100;
    cfg.sample_budget = 2.0e5;
    cfg.num_record_points = 160;
    cfg.repeats_per_pair = 50;
    cfg.bootstrap_repetitions = 1000;

    cfg.output_folder = 'results_radar_synthetic_multi_alpha';
    cfg.save_mat = true;
    cfg.save_partial_mat = true;
    cfg.save_fig = true;
    cfg.save_png = true;
    cfg.save_pdf = true;
    cfg.png_resolution = 600;
    cfg.keep_figures_open = true;

    cfg.alpha_values = [1.2, 1.5, 1.8];
    cfg.dispersion_values_m = [10, 20, 30, 40, 50];
    cfg.reference_alpha = 1.5;
    cfg.reference_dispersion_m = 30;

    cfg.length_scale_m = 600;
    cfg.rho_m = 0.10;
    cfg.outer_delta = 1.0;

    % NSCG-M: one update costs 50+50=100 measurement evaluations.
    cfg.alg.NSCG_M.eta0 = 1.0e-2;
    cfg.alg.NSCG_M.g_batch = 50;
    cfg.alg.NSCG_M.J_batch = 50;
    cfg.alg.NSCG_M.decay_start_fraction = 0.05;
    cfg.alg.NSCG_M.decay_power = 1.00;

    % NSCG-S: one four-update cycle costs
    %   (128+128)+3*(24+24)=400,
    % hence the average cost is also 100 evaluations per update.
    cfg.alg.NSCG_S.eta0 = 1.0e-2;
    cfg.alg.NSCG_S.tau = 4;
    cfg.alg.NSCG_S.outer_g_batch = 128;
    cfg.alg.NSCG_S.outer_J_batch = 128;
    cfg.alg.NSCG_S.inner_g_batch = 24;
    cfg.alg.NSCG_S.inner_J_batch = 24;
    cfg.alg.NSCG_S.decay_start_fraction = 0.10;
    cfg.alg.NSCG_S.decay_power = 2.00;

    cfg.method_order = {'NSCG_M', 'NSCG_S'};
    cfg.method_labels.NSCG_M = 'NSCG-M';
    cfg.method_labels.NSCG_S = 'NSCG-S';

    cfg.divergence_radius_normalized = 1.0e4;
    cfg.divergence_value_m = 1.0e9;
    cfg.min_positive_y = 1.0e-8;
    cfg.success_thresholds_m = [5, 10];

    cfg.figure_position = [100, 100, 920, 640];

    cfg.robustness_figure_position = [100, 40, 920, 900];
    cfg.robustness_alpha_order = [1.8, 1.5, 1.2];
    cfg.robustness_axes_positions = [ ...
        0.12, 0.690, 0.83, 0.190; ...
        0.12, 0.405, 0.83, 0.190; ...
        0.12, 0.120, 0.83, 0.190];
    cfg.robustness_legend_position = [0.355, 0.905, 0.290, 0.038];
    cfg.geometry_figure_position = [80, 80, 1060, 650];
    cfg.font_size = 16;
    cfg.legend_font_size = 12;
    cfg.line_width = 2.5;
    cfg.marker_size = 7;
    cfg.number_of_markers = 9;
    cfg.legend_location = 'best';

    cfg.plot_styles.NSCG_M = struct( ...
        'color', [0.00 0.62 0.45], ...
        'line', '--', 'marker', '>', ...
        'marker_face', [0.00 0.62 0.45]);
    cfg.plot_styles.NSCG_S = struct( ...
        'color', [0.84 0.15 0.16], ...
        'line', '-', 'marker', 'p', ...
        'marker_face', [0.84 0.15 0.16]);

    cfg.geometry_ellipse_pair_ids = [16, 20, 25, 54];
    cfg.geometry_main_axes_position = [0.07, 0.11, 0.60, 0.80];
    cfg.geometry_inset_axes_position = [0.70, 0.53, 0.27, 0.35];
    cfg.geometry_legend_position = [0.70, 0.12, 0.27, 0.28];
    cfg.geometry_zoom_margin_fraction = 0.12;
    cfg.geometry_arrow_count = 3;
end

function validate_config(cfg)
    assert(cfg.num_runs >= 1 && cfg.num_runs == floor(cfg.num_runs), ...
        'num_runs must be a positive integer.');
    assert(cfg.sample_budget > 0 && isfinite(cfg.sample_budget), ...
        'sample_budget must be positive and finite.');
    assert(cfg.repeats_per_pair >= 1 && ...
        cfg.repeats_per_pair == floor(cfg.repeats_per_pair), ...
        'repeats_per_pair must be a positive integer.');
    assert(isvector(cfg.alpha_values) && ~isempty(cfg.alpha_values) && ...
        all(isfinite(cfg.alpha_values)) && ...
        all(cfg.alpha_values > 1) && all(cfg.alpha_values < 2), ...
        'Every alpha value must lie in (1,2).');
    assert(isvector(cfg.dispersion_values_m) && ...
        ~isempty(cfg.dispersion_values_m) && ...
        all(isfinite(cfg.dispersion_values_m)) && ...
        all(cfg.dispersion_values_m > 0), ...
        'Every dispersion value must be positive.');
    assert(cfg.length_scale_m > 0 && cfg.rho_m > 0, ...
        'length_scale_m and rho_m must be positive.');
    assert(any(abs(cfg.alpha_values-cfg.reference_alpha) <= ...
        100*eps(max(1,abs(cfg.reference_alpha)))), ...
        'reference_alpha must occur in alpha_values.');
    assert(any(abs(cfg.dispersion_values_m-cfg.reference_dispersion_m) <= ...
        100*eps(max(1,abs(cfg.reference_dispersion_m)))), ...
        'reference_dispersion_m must occur in dispersion_values_m.');

    validate_algorithm(cfg.alg.NSCG_M, false);
    validate_algorithm(cfg.alg.NSCG_S, true);

    cost_m = cfg.alg.NSCG_M.g_batch + cfg.alg.NSCG_M.J_batch;
    cost_s = (cfg.alg.NSCG_S.outer_g_batch + ...
        cfg.alg.NSCG_S.outer_J_batch + ...
        (cfg.alg.NSCG_S.tau-1)*(cfg.alg.NSCG_S.inner_g_batch + ...
        cfg.alg.NSCG_S.inner_J_batch)) / cfg.alg.NSCG_S.tau;
    assert(abs(cost_m-cost_s) <= 100*eps(max(cost_m,cost_s)), ...
        ['The default comparison requires matched average costs: ', ...
         'NSCG-M=%.6g, NSCG-S=%.6g.'], cost_m, cost_s);
end

function validate_algorithm(alg, spider)
    assert(alg.eta0 > 0 && alg.decay_start_fraction > 0 && ...
        alg.decay_start_fraction <= 1 && alg.decay_power > 0, ...
        'Invalid step-size schedule.');
    if spider
        assert(is_positive_integer(alg.tau) && alg.tau >= 2 && ...
            is_positive_integer(alg.outer_g_batch) && ...
            is_positive_integer(alg.outer_J_batch) && ...
            is_positive_integer(alg.inner_g_batch) && ...
            is_positive_integer(alg.inner_J_batch), ...
            'Invalid NSCG-S batch settings.');
    else
        assert(is_positive_integer(alg.g_batch) && ...
            is_positive_integer(alg.J_batch), ...
            'Invalid NSCG-M batch settings.');
    end
end

%% ========================================================================
% Literature geometry and data generation
% =========================================================================
function problem = build_literature_geometry(cfg)

    tx_m = [ ...
        -300, -200; ...
        -300,  200; ...
        -200, -300; ...
        -200,  300; ...
         200, -300; ...
         200,  300; ...
         300,  200; ...
         300, -200];

    rx_m = [ ...
        -450, -450; ...
        -450,  450; ...
         450, -450; ...
         600,    0; ...
           0,  600; ...
         450,  450; ...
           0, -600; ...
        -600,    0];

    problem = struct();
    problem.d = 2;
    problem.Nt = size(tx_m, 1);
    problem.Nr = size(rx_m, 1);
    problem.number_of_pairs = problem.Nt * problem.Nr;
    problem.tx_m = tx_m';
    problem.rx_m = rx_m';
    problem.x_star_m = [400; 100];
    problem.x0_m = [0; 0];
    problem.length_scale_m = cfg.length_scale_m;
    problem.rho_m = cfg.rho_m;

    problem.tx = problem.tx_m ./ cfg.length_scale_m;
    problem.rx = problem.rx_m ./ cfg.length_scale_m;
    problem.x_star = problem.x_star_m ./ cfg.length_scale_m;
    problem.x0 = problem.x0_m ./ cfg.length_scale_m;
    problem.rho = cfg.rho_m ./ cfg.length_scale_m;
    problem.outer_delta = cfg.outer_delta;
    problem.divergence_radius = cfg.divergence_radius_normalized;
    problem.divergence_value_m = cfg.divergence_value_m;

    pair_tx = zeros(problem.number_of_pairs, 1);
    pair_rx = zeros(problem.number_of_pairs, 1);
    cursor = 0;
    for receiver = 1:problem.Nr
        for transmitter = 1:problem.Nt
            cursor = cursor + 1;
            pair_tx(cursor) = transmitter;
            pair_rx(cursor) = receiver;
        end
    end
    problem.pair_tx = pair_tx;
    problem.pair_rx = pair_rx;

    h_star = bistatic_range_batch(problem, problem.x_star, ...
        pair_tx, pair_rx);
    problem.h_star = h_star(:);

    [~, J_star] = exact_population_map(problem, problem.x_star);
    eigenvalues = eig(0.5*(J_star+J_star'));
    if min(eigenvalues) <= 1.0e-8
        warning('main_radar_synthetic:WeakGeometry', ...
            'The population Jacobian at x_star is nearly singular.');
    end
end

function frame = generate_measurement_frame( ...
        problem, cfg, alpha, dispersion_m, seed)
    rng(seed, 'twister');

    number_of_observations = ...
        problem.number_of_pairs * cfg.repeats_per_pair;
    pair_ids = repmat((1:problem.number_of_pairs)', ...
        cfg.repeats_per_pair, 1);

    epsilon_m = draw_symmetric_alpha_stable( ...
        number_of_observations, alpha, dispersion_m);
    z_m = problem.h_star(pair_ids) .* cfg.length_scale_m + epsilon_m;

    permutation = randperm(number_of_observations);
    pair_ids = pair_ids(permutation);
    z_m = z_m(permutation);

    frame = struct();
    frame.number_of_observations = number_of_observations;
    frame.pair_id = pair_ids;
    frame.tx_index = problem.pair_tx(pair_ids);
    frame.rx_index = problem.pair_rx(pair_ids);
    frame.z = z_m ./ cfg.length_scale_m;
    frame.alpha = alpha;
    frame.dispersion_m = dispersion_m;
end

function values = draw_symmetric_alpha_stable(n, alpha, sigma)
    assert(alpha > 1 && alpha < 2, 'alpha must lie in (1,2).');
    assert(sigma > 0, 'sigma must be positive.');

    values = zeros(n, 1);
    unfinished = true(n, 1);
    while any(unfinished)
        count = sum(unfinished);
        U = pi .* (rand(count, 1) - 0.5);
        W = -log(max(rand(count, 1), realmin('double')));
        standard_values = sin(alpha .* U) ./ ...
            (cos(U).^(1 / alpha)) .* ...
            (cos((1 - alpha) .* U) ./ W).^ ...
            ((1 - alpha) / alpha);
        candidate = sigma .* standard_values;
        finite_mask = isfinite(candidate);
        indices = find(unfinished);
        values(indices(finite_mask)) = candidate(finite_mask);
        unfinished(indices(finite_mask)) = false;
    end
end

%% ========================================================================
% Geometry, stochastic oracles, and population metrics
% =========================================================================
function h = bistatic_range_batch(problem, x, tx_index, rx_index)
    tx = problem.tx(:, tx_index);
    rx = problem.rx(:, rx_index);
    u_tx = bsxfun(@minus, x, tx);
    u_rx = bsxfun(@minus, x, rx);
    h = sqrt(sum(u_tx.^2, 1) + problem.rho^2) + ...
        sqrt(sum(u_rx.^2, 1) + problem.rho^2);
end

function [h, gradient, Hessian] = bistatic_geometry_batch( ...
        problem, x, tx_index, rx_index)
    tx = problem.tx(:, tx_index);
    rx = problem.rx(:, rx_index);
    u_tx = bsxfun(@minus, x, tx);
    u_rx = bsxfun(@minus, x, rx);

    r_tx = sqrt(sum(u_tx.^2, 1) + problem.rho^2);
    r_rx = sqrt(sum(u_rx.^2, 1) + problem.rho^2);
    h = r_tx + r_rx;
    gradient = bsxfun(@rdivide, u_tx, r_tx) + ...
        bsxfun(@rdivide, u_rx, r_rx);

    n = numel(tx_index);
    inverse_tx = 1 ./ r_tx;
    inverse_rx = 1 ./ r_rx;
    inverse_tx3 = inverse_tx.^3;
    inverse_rx3 = inverse_rx.^3;
    Hessian = zeros(2, 2, n);

    Hessian(1, 1, :) = reshape( ...
        inverse_tx - u_tx(1, :).^2 .* inverse_tx3 + ...
        inverse_rx - u_rx(1, :).^2 .* inverse_rx3, 1, 1, n);
    Hessian(2, 2, :) = reshape( ...
        inverse_tx - u_tx(2, :).^2 .* inverse_tx3 + ...
        inverse_rx - u_rx(2, :).^2 .* inverse_rx3, 1, 1, n);
    off_diagonal = -u_tx(1, :) .* u_tx(2, :) .* inverse_tx3 - ...
        u_rx(1, :) .* u_rx(2, :) .* inverse_rx3;
    Hessian(1, 2, :) = reshape(off_diagonal, 1, 1, n);
    Hessian(2, 1, :) = reshape(off_diagonal, 1, 1, n);
end

function g_hat = estimate_g_batch(problem, frame, x, batch_size)
    sample_ids = randi(frame.number_of_observations, batch_size, 1);
    tx_index = frame.tx_index(sample_ids);
    rx_index = frame.rx_index(sample_ids);
    [h, gradient] = bistatic_geometry_batch( ...
        problem, x, tx_index, rx_index);
    residual = h - frame.z(sample_ids)';
    g_hat = mean(bsxfun(@times, gradient, residual), 2);
end

function J_hat = estimate_J_batch(problem, frame, x, batch_size)
    sample_ids = randi(frame.number_of_observations, batch_size, 1);
    tx_index = frame.tx_index(sample_ids);
    rx_index = frame.rx_index(sample_ids);
    [h, gradient, Hessian] = bistatic_geometry_batch( ...
        problem, x, tx_index, rx_index);
    residual = h - frame.z(sample_ids)';

    gram_term = (gradient * gradient') ./ batch_size;
    hessian_term = mean(bsxfun(@times, Hessian, ...
        reshape(residual, 1, 1, batch_size)), 3);
    J_hat = gram_term + hessian_term;
end

function difference = estimate_g_difference( ...
        problem, frame, x_new, x_old, batch_size)
    sample_ids = randi(frame.number_of_observations, batch_size, 1);
    tx_index = frame.tx_index(sample_ids);
    rx_index = frame.rx_index(sample_ids);
    [h_new, gradient_new] = bistatic_geometry_batch( ...
        problem, x_new, tx_index, rx_index);
    [h_old, gradient_old] = bistatic_geometry_batch( ...
        problem, x_old, tx_index, rx_index);
    z = frame.z(sample_ids)';
    g_new = bsxfun(@times, gradient_new, h_new - z);
    g_old = bsxfun(@times, gradient_old, h_old - z);
    difference = mean(g_new - g_old, 2);
end

function difference = estimate_J_difference( ...
        problem, frame, x_new, x_old, batch_size)
    sample_ids = randi(frame.number_of_observations, batch_size, 1);
    tx_index = frame.tx_index(sample_ids);
    rx_index = frame.rx_index(sample_ids);
    [h_new, gradient_new, Hessian_new] = bistatic_geometry_batch( ...
        problem, x_new, tx_index, rx_index);
    [h_old, gradient_old, Hessian_old] = bistatic_geometry_batch( ...
        problem, x_old, tx_index, rx_index);
    z = frame.z(sample_ids)';
    residual_new = h_new - z;
    residual_old = h_old - z;

    gram_difference = (gradient_new * gradient_new' - ...
        gradient_old * gradient_old') ./ batch_size;
    hessian_difference = mean( ...
        bsxfun(@times, Hessian_new, ...
            reshape(residual_new, 1, 1, batch_size)) - ...
        bsxfun(@times, Hessian_old, ...
            reshape(residual_old, 1, 1, batch_size)), 3);
    difference = gram_difference + hessian_difference;
end

function [G, J, range_residual] = exact_population_map(problem, x)
    [h, gradient, Hessian] = bistatic_geometry_batch( ...
        problem, x, problem.pair_tx, problem.pair_rx);
    range_residual = h(:) - problem.h_star;
    G = mean(bsxfun(@times, gradient, range_residual'), 2);
    J = (gradient * gradient') ./ problem.number_of_pairs + ...
        mean(bsxfun(@times, Hessian, ...
        reshape(range_residual, 1, 1, problem.number_of_pairs)), 3);
end

function gradient = compose_gradient(J_hat, g_hat, delta)
    gradient = J_hat' * radial_pseudo_huber_gradient(g_hat, delta);
end

function gradient = radial_pseudo_huber_gradient(u, delta)
    u_norm = stable_norm(u);
    if u_norm == 0
        gradient = zeros(size(u));
    elseif isfinite(u_norm)
        gradient = (delta / hypot(delta, u_norm)) .* u;
    else
        gradient = delta .* safe_normalize(u);
    end
end

%% ========================================================================
% NSCG-M and NSCG-S
% =========================================================================
function trajectory = run_nscg_m( ...
        problem, frame, alg, cfg, checkpoints, seed)
    rng(seed, 'twister');
    x = problem.x0;
    samples_used = 0;
    trajectory = initialize_trajectory(checkpoints, x, problem);
    step_cost = alg.g_batch + alg.J_batch;

    while samples_used + step_cost <= cfg.sample_budget
        g_hat = estimate_g_batch(problem, frame, x, alg.g_batch);
        J_hat = estimate_J_batch(problem, frame, x, alg.J_batch);
        gradient_hat = compose_gradient(J_hat, g_hat, problem.outer_delta);

        next_samples = samples_used + step_cost;
        eta = sample_based_step(alg, next_samples, cfg.sample_budget);
        x = x - eta .* safe_normalize(gradient_hat);
        samples_used = next_samples;

        trajectory = record_if_reached( ...
            trajectory, samples_used, x, problem);
        if has_diverged(x, problem)
            x(:) = NaN;
            break;
        end
    end
    trajectory = finalize_trajectory(trajectory, x, problem);
end

function trajectory = run_nscg_s( ...
        problem, frame, alg, cfg, checkpoints, seed)
    rng(seed, 'twister');
    x = problem.x0;
    samples_used = 0;
    trajectory = initialize_trajectory(checkpoints, x, problem);

    outer_cost = alg.outer_g_batch + alg.outer_J_batch;
    inner_cost = alg.inner_g_batch + alg.inner_J_batch;

    while samples_used + outer_cost <= cfg.sample_budget
        x_outer = x;
        g_hat = estimate_g_batch( ...
            problem, frame, x_outer, alg.outer_g_batch);
        J_hat = estimate_J_batch( ...
            problem, frame, x_outer, alg.outer_J_batch);
        samples_used = samples_used + outer_cost;

        eta = sample_based_step(alg, samples_used, cfg.sample_budget);
        gradient_hat = compose_gradient(J_hat, g_hat, problem.outer_delta);
        x_current = x_outer - eta .* safe_normalize(gradient_hat);
        trajectory = record_if_reached( ...
            trajectory, samples_used, x_current, problem);

        if has_diverged(x_current, problem)
            x = NaN(problem.d, 1);
            break;
        end

        x_previous = x_outer;
        for inner_index = 2:alg.tau %#ok<NASGU>
            if samples_used + inner_cost > cfg.sample_budget
                break;
            end

            g_hat = g_hat + estimate_g_difference( ...
                problem, frame, x_current, x_previous, ...
                alg.inner_g_batch);
            J_hat = J_hat + estimate_J_difference( ...
                problem, frame, x_current, x_previous, ...
                alg.inner_J_batch);
            samples_used = samples_used + inner_cost;

            eta = sample_based_step(alg, samples_used, cfg.sample_budget);
            gradient_hat = compose_gradient( ...
                J_hat, g_hat, problem.outer_delta);
            x_next = x_current - eta .* safe_normalize(gradient_hat);
            x_previous = x_current;
            x_current = x_next;

            trajectory = record_if_reached( ...
                trajectory, samples_used, x_current, problem);
            if has_diverged(x_current, problem)
                x_current(:) = NaN;
                break;
            end
        end

        x = x_current;
        if has_diverged(x, problem)
            x(:) = NaN;
            break;
        end
    end
    trajectory = finalize_trajectory(trajectory, x, problem);
end

function eta = sample_based_step(alg, samples_used, budget)
    start_sample = max(1, alg.decay_start_fraction .* budget);
    if samples_used <= start_sample
        factor = 1;
    else
        factor = (start_sample ./ samples_used) .^ alg.decay_power;
    end
    eta = alg.eta0 .* factor;
end


%% ========================================================================
% Recording, statistics, and seeds
% =========================================================================
function trajectory = initialize_trajectory(checkpoints, x, problem)
    number_of_points = numel(checkpoints);
    trajectory.samples = checkpoints;
    trajectory.location_error_m = NaN(1, number_of_points);
    trajectory.br_rmse_m = NaN(1, number_of_points);
    trajectory.stationarity = NaN(1, number_of_points);
    trajectory.position_m = NaN(problem.d, number_of_points);
    trajectory.next_index = 1;
    trajectory = record_if_reached(trajectory, 0, x, problem);
end

function trajectory = record_if_reached( ...
        trajectory, samples_used, x, problem)
    if trajectory.next_index > numel(trajectory.samples) || ...
            samples_used < trajectory.samples(trajectory.next_index)
        return;
    end
    metrics = exact_metrics(x, problem);
    while trajectory.next_index <= numel(trajectory.samples) && ...
            samples_used >= trajectory.samples(trajectory.next_index)
        index = trajectory.next_index;
        trajectory.location_error_m(index) = metrics.location_error_m;
        trajectory.br_rmse_m(index) = metrics.br_rmse_m;
        trajectory.stationarity(index) = metrics.stationarity;
        trajectory.position_m(:, index) = metrics.position_m;
        trajectory.next_index = trajectory.next_index + 1;
    end
end

function trajectory = finalize_trajectory(trajectory, x, problem)
    metrics = exact_metrics(x, problem);
    while trajectory.next_index <= numel(trajectory.samples)
        index = trajectory.next_index;
        trajectory.location_error_m(index) = metrics.location_error_m;
        trajectory.br_rmse_m(index) = metrics.br_rmse_m;
        trajectory.stationarity(index) = metrics.stationarity;
        trajectory.position_m(:, index) = metrics.position_m;
        trajectory.next_index = trajectory.next_index + 1;
    end
    trajectory = rmfield(trajectory, 'next_index');
end

function metrics = exact_metrics(x, problem)
    if any(~isfinite(x))
        metrics.location_error_m = problem.divergence_value_m;
        metrics.br_rmse_m = problem.divergence_value_m;
        metrics.stationarity = problem.divergence_value_m;
        metrics.position_m = NaN(problem.d, 1);
        return;
    end
    [G, J, residual] = exact_population_map(problem, x);
    gradient = compose_gradient(J, G, problem.outer_delta);
    metrics.position_m = x .* problem.length_scale_m;
    metrics.location_error_m = norm( ...
        metrics.position_m - problem.x_star_m);
    metrics.br_rmse_m = sqrt(mean(residual.^2)) .* ...
        problem.length_scale_m;
    metrics.stationarity = norm(gradient);
end

function raw = initialize_raw(cfg, dimension, number_of_points)
    raw = struct();
    for k = 1:numel(cfg.method_order)
        name = cfg.method_order{k};
        raw.(name).location_error_m = ...
            zeros(cfg.num_runs, number_of_points);
        raw.(name).br_rmse_m = zeros(cfg.num_runs, number_of_points);
        raw.(name).stationarity = zeros(cfg.num_runs, number_of_points);
        raw.(name).position_m = ...
            zeros(cfg.num_runs, dimension, number_of_points);
        raw.(name).runtime_s = zeros(cfg.num_runs, 1);
    end
end

function summary = aggregate_case(raw, checkpoints, cfg, problem)
    summary = struct();
    z_value = 1.959963984540054;
    for k = 1:numel(cfg.method_order)
        name = cfg.method_order{k};
        errors = raw.(name).location_error_m;
        mean_error = mean(errors, 1);
        if cfg.num_runs > 1
            se = std(errors, 0, 1) ./ sqrt(cfg.num_runs);
        else
            se = zeros(size(mean_error));
        end
        summary.(name).samples = checkpoints;
        summary.(name).mean_location_error_m = mean_error;
        summary.(name).ci_lower_m = max( ...
            mean_error - z_value .* se, cfg.min_positive_y);
        summary.(name).ci_upper_m = mean_error + z_value .* se;
        summary.(name).mean_br_rmse_m = mean(raw.(name).br_rmse_m, 1);
        summary.(name).mean_stationarity = mean(raw.(name).stationarity, 1);
        summary.(name).mean_runtime_s = mean(raw.(name).runtime_s);
        summary.(name).x_star_m = problem.x_star_m;
    end
end

function statistics = final_case_statistics( ...
        raw, cfg, problem, alpha, dispersion_m)
    statistics = struct();
    for k = 1:numel(cfg.method_order)
        name = cfg.method_order{k};
        final_errors = raw.(name).location_error_m(:, end);
        final_positions = squeeze(raw.(name).position_m(:, :, end));
        if size(final_positions, 2) ~= problem.d
            final_positions = reshape(final_positions, [], problem.d);
        end
        statistics.(name).rmse_m = sqrt(mean(final_errors.^2));
        bootstrap_seed = make_bootstrap_seed( ...
            cfg, alpha, dispersion_m, k);
        statistics.(name).rmse_ci95_m = bootstrap_rmse_ci( ...
            final_errors, cfg.bootstrap_repetitions, bootstrap_seed);
        statistics.(name).median_m = empirical_quantile(final_errors, 0.5);
        statistics.(name).q90_m = empirical_quantile(final_errors, 0.9);
        statistics.(name).bias_m = norm(mean(final_positions, 1)' - ...
            problem.x_star_m);
        statistics.(name).success_5m = mean(final_errors <= 5);
        statistics.(name).success_10m = mean(final_errors <= 10);
    end
end

function interval = bootstrap_rmse_ci(values, number_of_bootstraps, seed)
    values = values(isfinite(values));
    if isempty(values)
        interval = [NaN, NaN];
        return;
    end
    if numel(values) == 1 || number_of_bootstraps < 2
        value = sqrt(mean(values.^2));
        interval = [value, value];
        return;
    end

    old_state = rng;
    rng(seed, 'twister');
    n = numel(values);
    indices = randi(n, n, number_of_bootstraps);
    resampled = reshape(values(indices), n, number_of_bootstraps);
    bootstrap_rmse = sqrt(mean(resampled.^2, 1));
    interval = [empirical_quantile(bootstrap_rmse, 0.025), ...
        empirical_quantile(bootstrap_rmse, 0.975)];
    rng(old_state);
end

function seed = make_data_seed(cfg, alpha, dispersion_m, run)
    seed = bounded_seed(cfg.data_seed + ...
        1000000*round(100*alpha) + ...
        1000*round(10*dispersion_m) + run);
end

function seed = make_algorithm_seed(cfg, alpha, dispersion_m, run)
    seed = bounded_seed(cfg.base_seed + ...
        1000000*round(100*alpha) + ...
        1000*round(10*dispersion_m) + run);
end

function seed = make_bootstrap_seed(cfg, alpha, dispersion_m, method_index)
    seed = bounded_seed(cfg.bootstrap_seed + ...
        1000000*round(100*alpha) + ...
        1000*round(10*dispersion_m) + method_index);
end

function seed = bounded_seed(value)
    modulus = 2147483646;
    seed = mod(round(value)-1, modulus) + 1;
end

function checkpoints = make_record_checkpoints(budget, requested_points)
    requested_points = max(30, round(requested_points));
    early_points = max(15, round(0.40 * requested_points));
    late_points = max(15, requested_points - early_points);
    split_budget = max(10, round(0.10 * budget));
    early = round(logspace(0, log10(split_budget), early_points));
    late = round(linspace(split_budget, budget, late_points));
    checkpoints = unique([0, early, late, budget]);
end

function result = empty_case_result()
    result = struct('alpha', NaN, 'dispersion_m', NaN, ...
        'checkpoints', [], 'raw', struct(), ...
        'summary', struct(), 'statistics', struct());
end

function result = find_reference_case(case_results, alpha, dispersion_m)
    tolerance_alpha = 100*eps(max(1,abs(alpha)));
    tolerance_dispersion = 100*eps(max(1,abs(dispersion_m)));
    match = find(abs([case_results.alpha]-alpha) <= tolerance_alpha & ...
        abs([case_results.dispersion_m]-dispersion_m) <= ...
        tolerance_dispersion, 1, 'first');
    if isempty(match)
        error('The requested reference case is absent.');
    end
    result = case_results(match);
end

%% ========================================================================
% Plotting
% =========================================================================
function fig = plot_rmse_vs_dispersion(case_results, cfg)

    fig = new_figure(cfg.robustness_figure_position);
    alpha_order = cfg.robustness_alpha_order;
    number_of_panels = numel(alpha_order);
    axes_handles = gobjects(1, number_of_panels);
    legend_handles = gobjects(1, numel(cfg.method_order));
    panel_tags = {'(a)', '(b)', '(c)'};
    x = cfg.dispersion_values_m;

    for panel_index = 1:number_of_panels
        alpha = alpha_order(panel_index);
        ax = axes('Parent', fig, ...
            'Position', cfg.robustness_axes_positions(panel_index, :));
        hold(ax, 'on');
        axes_handles(panel_index) = ax;

        for method_index = 1:numel(cfg.method_order)
            method_name = cfg.method_order{method_index};
            style = cfg.plot_styles.(method_name);
            y = NaN(size(x));
            lower = NaN(size(x));
            upper = NaN(size(x));

            for dispersion_index = 1:numel(x)
                one_case = find_reference_case( ...
                    case_results, alpha, x(dispersion_index));
                stats = one_case.statistics.(method_name);
                y(dispersion_index) = stats.rmse_m;
                lower(dispersion_index) = stats.rmse_ci95_m(1);
                upper(dispersion_index) = stats.rmse_ci95_m(2);
            end

            y = max(y, cfg.min_positive_y);
            lower = max(lower, cfg.min_positive_y);
            upper = max(upper, lower);
            draw_vertical_error_bars(ax, x, lower, upper, style.color);

            curve_handle = semilogy(ax, x, y, ...
                'Color', style.color, 'LineStyle', style.line, ...
                'LineWidth', cfg.line_width, 'Marker', style.marker, ...
                'MarkerSize', cfg.marker_size, ...
                'MarkerFaceColor', style.marker_face, ...
                'MarkerEdgeColor', style.color);
            if panel_index == 1
                legend_handles(method_index) = curve_handle;
            else
                set(curve_handle, 'HandleVisibility', 'off');
            end
        end

        format_axes(ax, cfg);
        xlim(ax, [min(x), max(x)]);
        set(ax, 'XTick', x);
        ylabel(ax, 'Final localization RMSE (m)', ...
            'Interpreter', 'none', 'FontSize', cfg.font_size);
        text(ax, 0.015, 0.86, ...
            [panel_tags{panel_index}, '  \alpha = ', ...
             num2str(alpha, '%.1f')], ...
            'Units', 'normalized', 'Interpreter', 'tex', ...
            'FontName', 'Times New Roman', ...
            'FontSize', cfg.font_size, 'FontWeight', 'normal', ...
            'VerticalAlignment', 'top', 'BackgroundColor', 'w');

        if panel_index < number_of_panels
            set(ax, 'XTickLabel', []);
        else
            xlabel(ax, 'S-alpha-S dispersion (m)', ...
                'Interpreter', 'none', 'FontSize', cfg.font_size + 1);
        end
    end

    annotation(fig, 'textbox', [0.08, 0.955, 0.84, 0.035], ...
        'String', 'Localization robustness under impulsive noise', ...
        'Interpreter', 'none', 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle', 'EdgeColor', 'none', ...
        'FontName', 'Times New Roman', ...
        'FontWeight', 'normal', 'FontSize', cfg.font_size + 3);

    legend_handle = legend(axes_handles(1), legend_handles, ...
        method_label_list(cfg), 'Orientation', 'horizontal', ...
        'Interpreter', 'none', 'FontSize', cfg.legend_font_size, ...
        'Box', 'off');
    set(legend_handle, 'Units', 'normalized', ...
        'Position', cfg.robustness_legend_position);
    if isprop(legend_handle, 'ItemTokenSize')
        legend_handle.ItemTokenSize = [30, 16];
    end
    linkaxes(axes_handles, 'x');
end

function draw_vertical_error_bars(ax, x, lower, upper, color)
    for k = 1:numel(x)
        if ~isfinite(lower(k)) || ~isfinite(upper(k))
            continue;
        end
        plot(ax, [x(k), x(k)], [lower(k), upper(k)], ...
            'Color', color, 'LineStyle', '-', ...
            'LineWidth', 0.9, 'HandleVisibility', 'off');
        half_cap = 0.012 * max(x(end)-x(1), 1);
        plot(ax, x(k)+[-half_cap, half_cap], [lower(k), lower(k)], ...
            'Color', color, 'LineWidth', 0.9, ...
            'HandleVisibility', 'off');
        plot(ax, x(k)+[-half_cap, half_cap], [upper(k), upper(k)], ...
            'Color', color, 'LineWidth', 0.9, ...
            'HandleVisibility', 'off');
    end
end

function fig = plot_geometry_and_paths(case_result, problem, cfg)
    fig = new_figure(cfg.geometry_figure_position);
    ax = axes('Parent', fig, ...
        'Position', cfg.geometry_main_axes_position);
    hold(ax, 'on');

    ellipse_color = [0.78, 0.80, 0.83];
    for k = 1:numel(cfg.geometry_ellipse_pair_ids)
        pair_id = cfg.geometry_ellipse_pair_ids(k);
        if pair_id >= 1 && pair_id <= problem.number_of_pairs
            tx = problem.tx_m(:, problem.pair_tx(pair_id));
            rx = problem.rx_m(:, problem.pair_rx(pair_id));
            draw_bistatic_ellipse(ax, tx, rx, ...
                problem.x_star_m, ellipse_color);
        end
    end

    tx_handle = plot(ax, problem.tx_m(1, :), problem.tx_m(2, :), 'o', ...
        'Color', [0.90 0.40 0.00], 'MarkerFaceColor', 'w', ...
        'MarkerSize', 7, 'LineWidth', 1.2);
    rx_handle = plot(ax, problem.rx_m(1, :), problem.rx_m(2, :), '^', ...
        'Color', [0.00 0.45 0.70], 'MarkerFaceColor', 'w', ...
        'MarkerSize', 7, 'LineWidth', 1.2);

    representative_run = common_representative_run(case_result, cfg);
    paths = cell(1, numel(cfg.method_order));
    method_handles = gobjects(1, numel(cfg.method_order));

    plot(ax, [problem.x0_m(1), problem.x_star_m(1)], ...
        [problem.x0_m(2), problem.x_star_m(2)], ...
        'Color', [0.72 0.72 0.72], 'LineStyle', ':', ...
        'LineWidth', 1.0, 'HandleVisibility', 'off');

    for k = 1:numel(cfg.method_order)
        name = cfg.method_order{k};
        style = cfg.plot_styles.(name);
        path = squeeze(case_result.raw.(name).position_m( ...
            representative_run, :, :));
        if size(path, 1) ~= problem.d
            path = reshape(path, problem.d, []);
        end
        paths{k} = path;
        plot_path_2d(ax, path, style, cfg, true);
        add_path_arrows_2d(ax, path, style.color, ...
            cfg.geometry_arrow_count);
        method_handles(k) = plot(ax, NaN, NaN, ...
            'Color', style.color, 'LineStyle', style.line, ...
            'LineWidth', cfg.line_width, 'Marker', style.marker, ...
            'MarkerSize', cfg.marker_size, ...
            'MarkerFaceColor', style.marker_face, ...
            'MarkerEdgeColor', style.color);
    end

    initial_handle = plot(ax, problem.x0_m(1), problem.x0_m(2), 'kx', ...
        'MarkerSize', 11, 'LineWidth', 2.1);
    target_handle = plot(ax, problem.x_star_m(1), problem.x_star_m(2), ...
        'kp', 'MarkerSize', 15, ...
        'MarkerFaceColor', [0.95 0.75 0.00], 'LineWidth', 1.3);

    format_axes(ax, cfg);
    axis(ax, 'equal');
    xlabel(ax, '$x_1$ (m)', 'Interpreter', 'latex', ...
        'FontSize', cfg.font_size + 2);
    ylabel(ax, '$x_2$ (m)', 'Interpreter', 'latex', ...
        'FontSize', cfg.font_size + 2);
    title(ax, {'Bistatic geometry and representative localization paths', ...
        ['alpha=', num2str(case_result.alpha,'%.1f'), ...
         ', dispersion=', num2str(case_result.dispersion_m,'%.3g'), ' m']}, ...
        'Interpreter', 'none', 'FontWeight', 'normal', ...
        'FontSize', cfg.font_size + 2);

    all_x = [problem.tx_m(1,:), problem.rx_m(1,:)];
    all_y = [problem.tx_m(2,:), problem.rx_m(2,:)];
    span = max([range_no_toolbox(all_x), range_no_toolbox(all_y), 1]);
    margin = 0.08*span;
    xlim(ax, [min(all_x)-margin, max(all_x)+margin]);
    ylim(ax, [min(all_y)-margin, max(all_y)+margin]);

    [zoom_x, zoom_y] = geometry_zoom_limits(paths, problem, ...
        cfg.geometry_zoom_margin_fraction);
    rectangle('Parent', ax, 'Position', ...
        [zoom_x(1), zoom_y(1), diff(zoom_x), diff(zoom_y)], ...
        'EdgeColor', [0.28 0.28 0.28], 'LineStyle', '--', ...
        'LineWidth', 1.0);

    inset = axes('Parent', fig, ...
        'Position', cfg.geometry_inset_axes_position);
    hold(inset, 'on');
    plot(inset, [problem.x0_m(1), problem.x_star_m(1)], ...
        [problem.x0_m(2), problem.x_star_m(2)], ...
        'Color', [0.78 0.78 0.78], 'LineStyle', ':', ...
        'LineWidth', 1.0, 'HandleVisibility', 'off');
    for k = 1:numel(cfg.method_order)
        name = cfg.method_order{k};
        style = cfg.plot_styles.(name);
        plot_path_2d(inset, paths{k}, style, cfg, true);
        add_path_arrows_2d(inset, paths{k}, style.color, ...
            cfg.geometry_arrow_count);
    end
    plot(inset, problem.x0_m(1), problem.x0_m(2), 'kx', ...
        'MarkerSize', 9, 'LineWidth', 1.8);
    plot(inset, problem.x_star_m(1), problem.x_star_m(2), 'kp', ...
        'MarkerSize', 13, 'MarkerFaceColor', [0.95 0.75 0.00], ...
        'LineWidth', 1.2);
    format_axes(inset, cfg);
    axis(inset, 'equal');
    xlim(inset, zoom_x);
    ylim(inset, zoom_y);
    title(inset, 'Trajectory zoom', 'Interpreter', 'none', ...
        'FontWeight', 'normal', 'FontSize', cfg.font_size - 1);
    set(inset, 'FontSize', cfg.font_size - 2);

    labels = [{'Transmitters', 'Receivers', 'Initial point', ...
        'True target'}, method_label_list(cfg)];
    legend_handle = legend(ax, [tx_handle, rx_handle, initial_handle, ...
        target_handle, method_handles], labels, ...
        'Interpreter', 'none', 'FontSize', cfg.legend_font_size);
    set(legend_handle, 'Units', 'normalized');
    set(legend_handle, 'Position', cfg.geometry_legend_position);
    if isprop(legend_handle, 'ItemTokenSize')
        legend_handle.ItemTokenSize = [26, 16];
    end
end

function draw_bistatic_ellipse(ax, focus_1, focus_2, target, color)
    total_range = norm(target-focus_1) + norm(target-focus_2);
    center = 0.5*(focus_1+focus_2);
    focal_half_distance = 0.5*norm(focus_2-focus_1);
    major_radius = 0.5*total_range;
    if major_radius <= focal_half_distance
        return;
    end
    minor_radius = sqrt(max(major_radius^2-focal_half_distance^2, 0));
    angle = atan2(focus_2(2)-focus_1(2), ...
        focus_2(1)-focus_1(1));
    theta = linspace(0,2*pi,500);
    local_points = [major_radius*cos(theta); minor_radius*sin(theta)];
    rotation = [cos(angle),-sin(angle);sin(angle),cos(angle)];
    points = bsxfun(@plus, rotation*local_points, center);
    plot(ax, points(1,:), points(2,:), ...
        'Color', color, 'LineWidth', 0.8, ...
        'LineStyle', '-', 'HandleVisibility', 'off');
end

function run = common_representative_run(case_result, cfg)
    errors_m = case_result.raw.NSCG_M.location_error_m(:,end);
    errors_s = case_result.raw.NSCG_S.location_error_m(:,end);
    scale_m = max(empirical_quantile(errors_m,0.5), eps);
    scale_s = max(empirical_quantile(errors_s,0.5), eps);
    score = 0.5*(errors_m./scale_m + errors_s./scale_s);
    target_score = empirical_quantile(score,0.5);
    [~,run] = min(abs(score-target_score));
    if run < 1 || run > cfg.num_runs
        run = 1;
    end
end

function [x_limits, y_limits] = geometry_zoom_limits(paths, problem, fraction)
    x_values = [problem.x0_m(1), problem.x_star_m(1)];
    y_values = [problem.x0_m(2), problem.x_star_m(2)];
    for k = 1:numel(paths)
        path = paths{k};
        finite = all(isfinite(path),1);
        x_values = [x_values, path(1,finite)]; %#ok<AGROW>
        y_values = [y_values, path(2,finite)]; %#ok<AGROW>
    end
    x_span = max(max(x_values)-min(x_values), 1);
    y_span = max(max(y_values)-min(y_values), 1);
    x_limits = [min(x_values)-fraction*x_span, ...
        max(x_values)+fraction*x_span];
    y_limits = [min(y_values)-fraction*y_span, ...
        max(y_values)+fraction*y_span];
end

function plot_path_2d(ax, path, style, cfg, show_markers)
    finite = all(isfinite(path),1);
    path = path(:,finite);
    if isempty(path)
        return;
    end
    plot(ax, path(1,:), path(2,:), ...
        'Color', style.color, 'LineStyle', style.line, ...
        'LineWidth', cfg.line_width, 'HandleVisibility', 'off');
    if show_markers
        marker_indices = unique(round(linspace(1,size(path,2), ...
            min(cfg.number_of_markers,size(path,2)))));
        plot(ax, path(1,marker_indices), path(2,marker_indices), ...
            'LineStyle', 'none', 'Marker', style.marker, ...
            'Color', style.color, 'MarkerSize', cfg.marker_size, ...
            'MarkerFaceColor', style.marker_face, ...
            'MarkerEdgeColor', style.color, 'HandleVisibility', 'off');
    end
end

function add_path_arrows_2d(ax, path, color, number_of_arrows)
    finite = all(isfinite(path),1);
    path = path(:,finite);
    if size(path,2) < 3 || number_of_arrows < 1
        return;
    end
    candidates = unique(round(linspace(2,size(path,2)-1, ...
        number_of_arrows)));
    for k = 1:numel(candidates)
        i = candidates(k);
        delta = path(:,i+1)-path(:,i);
        if norm(delta) <= eps
            continue;
        end
        quiver(ax, path(1,i), path(2,i), delta(1), delta(2), 0, ...
            'Color', color, 'LineWidth', 1.0, ...
            'MaxHeadSize', 1.2, 'HandleVisibility', 'off');
    end
end

function fig = plot_final_scatter(case_result, problem, cfg)
    fig = new_figure(cfg.figure_position);
    ax = axes(fig);
    hold(ax, 'on');

    target_handle = plot(ax, problem.x_star_m(1), problem.x_star_m(2), ...
        'kp', 'MarkerSize', 14, ...
        'MarkerFaceColor', [0.95 0.75 0.00], 'LineWidth', 1.2);
    method_handles = gobjects(1, numel(cfg.method_order));
    radial_limit = 10;
    outside_total = 0;

    stored_positions = cell(1,numel(cfg.method_order));
    stored_errors = cell(1,numel(cfg.method_order));
    for k = 1:numel(cfg.method_order)
        name = cfg.method_order{k};
        positions = squeeze(case_result.raw.(name).position_m(:,:,end));
        if size(positions,2) ~= problem.d
            positions = reshape(positions,[],problem.d);
        end
        stored_positions{k} = positions;
        errors = sqrt(sum(bsxfun(@minus, positions, ...
            problem.x_star_m').^2,2));
        stored_errors{k} = errors;
        q90 = empirical_quantile(errors,0.90);
        radial_limit = max(radial_limit,1.35*q90);
    end

    for k = 1:numel(cfg.method_order)
        name = cfg.method_order{k};
        style = cfg.plot_styles.(name);
        positions = stored_positions{k};
        errors = stored_errors{k};
        visible = errors <= radial_limit;
        outside_total = outside_total + sum(~visible & isfinite(errors));
        method_handles(k) = plot(ax, positions(visible,1), ...
            positions(visible,2), 'LineStyle', 'none', ...
            'Marker', style.marker, 'Color', style.color, ...
            'MarkerSize', cfg.marker_size, ...
            'MarkerFaceColor', style.marker_face, ...
            'MarkerEdgeColor', style.color);
        q90 = empirical_quantile(errors,0.90);
        draw_circle(ax, problem.x_star_m, q90, ...
            style.color, style.line);
    end

    format_axes(ax,cfg);
    axis(ax,'equal');
    xlim(ax,problem.x_star_m(1)+[-radial_limit,radial_limit]);
    ylim(ax,problem.x_star_m(2)+[-radial_limit,radial_limit]);
    xlabel(ax,'$x_1$ (m)','Interpreter','latex', ...
        'FontSize',cfg.font_size+2);
    ylabel(ax,'$x_2$ (m)','Interpreter','latex', ...
        'FontSize',cfg.font_size+2);
    title(ax,['Final estimates and 90% error circles, alpha=', ...
        num2str(case_result.alpha,'%.1f'), ', dispersion=', ...
        num2str(case_result.dispersion_m,'%.3g'),' m'], ...
        'Interpreter','none','FontWeight','normal', ...
        'FontSize',cfg.font_size+2);
    legend(ax,[target_handle,method_handles], ...
        [{'True target'},method_label_list(cfg)], ...
        'Location',cfg.legend_location,'Interpreter','none', ...
        'FontSize',cfg.legend_font_size);
    if outside_total > 0
        text(ax,0.02,0.03,[num2str(outside_total), ...
            ' estimates outside display window'], ...
            'Units','normalized','FontSize',cfg.legend_font_size, ...
            'BackgroundColor','w','EdgeColor',[0.7 0.7 0.7]);
    end
end

function fig = new_figure(position)
    fig = figure('Position', position, 'Color', 'w');
end

function format_axes(ax, cfg)
    grid(ax, 'on');
    box(ax, 'on');
    set(ax, 'FontName', 'Times New Roman', ...
        'FontSize', cfg.font_size, 'LineWidth', 1.1);
    if isprop(ax, 'GridAlpha')
        ax.GridAlpha = 0.20;
    end
    if isprop(ax, 'MinorGridAlpha')
        ax.MinorGridAlpha = 0.10;
    end
end

function labels = method_label_list(cfg)
    labels = cell(1, numel(cfg.method_order));
    for k = 1:numel(cfg.method_order)
        labels{k} = cfg.method_labels.(cfg.method_order{k});
    end
end

function draw_circle(ax, center, radius, color, line_style)
    if ~isfinite(radius) || radius <= 0
        return;
    end
    theta = linspace(0, 2*pi, 300);
    plot(ax, center(1)+radius*cos(theta), ...
        center(2)+radius*sin(theta), ...
        'Color', color, 'LineStyle', line_style, ...
        'LineWidth', 1.1, 'HandleVisibility', 'off');
end

function q = empirical_quantile(values, probability)
    values = sort(values(isfinite(values)));
    if isempty(values)
        q = NaN;
        return;
    end
    probability = min(max(probability,0),1);
    position = 1+(numel(values)-1)*probability;
    lower_index = floor(position);
    upper_index = ceil(position);
    if lower_index == upper_index
        q = values(lower_index);
    else
        weight = position-lower_index;
        q = (1-weight)*values(lower_index)+weight*values(upper_index);
    end
end

function value = range_no_toolbox(values)
    finite_values = values(isfinite(values));
    if isempty(finite_values)
        value = 0;
    else
        value = max(finite_values)-min(finite_values);
    end
end

function name = make_reference_name_2d(prefix, alpha, dispersion_m)
    alpha_tag = strrep(num2str(alpha,'%.3g'),'.','p');
    dispersion_tag = strrep(num2str(dispersion_m,'%.6g'),'.','p');
    name = [prefix,'_alpha_',alpha_tag,'_gamma_',dispersion_tag];
end

function save_figure_files(fig, base_path, cfg)
    if cfg.save_fig
        savefig(fig, [base_path, '.fig']);
    end

    old_units = get(fig, 'Units');
    set(fig, 'Units', 'inches');
    figure_position = get(fig, 'Position');
    set(fig, 'PaperUnits', 'inches', ...
        'PaperPositionMode', 'manual', ...
        'PaperPosition', [0, 0, figure_position(3), figure_position(4)], ...
        'PaperSize', [figure_position(3), figure_position(4)]);
    set(fig, 'Units', old_units);

    if cfg.save_png
        if exist('exportgraphics', 'file') == 2
            exportgraphics(fig, [base_path, '.png'], ...
                'Resolution', cfg.png_resolution);
        else
            print(fig, [base_path, '.png'], '-dpng', ...
                sprintf('-r%d', cfg.png_resolution));
        end
    end
    if cfg.save_pdf
        if exist('exportgraphics', 'file') == 2
            exportgraphics(fig, [base_path, '.pdf'], ...
                'ContentType', 'vector');
        else
            print(fig, [base_path, '.pdf'], ...
                '-dpdf', '-painters', '-bestfit');
        end
    end
end

function close_if_requested(fig, cfg)
    if ~cfg.keep_figures_open
        close(fig);
    end
end

%% ========================================================================
% Numerical safeguards
% =========================================================================
function direction = safe_normalize(vector)
    if isempty(vector)
        direction = vector;
        return;
    end
    infinite_mask = isinf(vector);
    if any(infinite_mask)
        direction = zeros(size(vector));
        direction(infinite_mask) = sign(vector(infinite_mask)) ./ ...
            sqrt(sum(infinite_mask));
        return;
    end
    scale = max(abs(vector));
    if isempty(scale) || scale == 0 || ~isfinite(scale)
        direction = zeros(size(vector));
        return;
    end
    scaled = vector ./ scale;
    scaled_norm = norm(scaled);
    if scaled_norm == 0 || ~isfinite(scaled_norm)
        direction = zeros(size(vector));
    else
        direction = scaled ./ scaled_norm;
    end
end

function value = stable_norm(vector)
    if any(isinf(vector))
        value = inf;
        return;
    end
    scale = max(abs(vector));
    if isempty(scale) || scale == 0
        value = 0;
    elseif ~isfinite(scale)
        value = inf;
    else
        value = scale .* norm(vector ./ scale);
    end
end

function flag = has_diverged(x, problem)
    flag = any(~isfinite(x)) || max(abs(x)) > problem.divergence_radius;
end

function flag = is_positive_integer(value)
    flag = isscalar(value) && isnumeric(value) && isfinite(value) && ...
        value >= 1 && value == floor(value);
end
