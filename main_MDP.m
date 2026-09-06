function main_MDP()
%   It compares seven methods under the same stochastic-sample budget:
%   ASC-PG, CIVR, SCSC, MVRC, Clipped-SCG, NSCG-M, and NSCG-S.
%
% Main implementation choices:
%   1. The outer objective is the mean pseudo-Huber Bellman loss
%          F(G(w)) = (1/S) sum_s rho_delta(phi_s'w-q_s^pi(w)),
%      where rho_delta(r)=delta^2(sqrt(1+(r/delta)^2)-1).
%   2. For each noise family, tail index, and run, one heavy-tailed reward table is generated. All algorithms in that run
%      use exactly the same MDP and the same w_star.
%   3. Within a fixed case and run, all seven algorithms are restarted from exactly the same random seed. 
%      Different runs use different seeds.
%   4. One sampled state index counts as one stochastic sample. When the same state sample is reused in a paired difference 
%      at two iterates, it is counted once rather than twice.

    clc;
    close all;

    cfg = default_config();

    project_root = fileparts(mfilename('fullpath'));
    output_dir = fullfile(project_root, cfg.output_folder);
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    validate_config(cfg);
    base_mdp = build_base_mdp(cfg);

    fprintf('============================================================\n');
    fprintf('Heavy-tailed stochastic compositional experiment\n');
    fprintf('S=%d, A=%d, successors=%d, gamma=%.2f\n', ...
        cfg.S, cfg.A, cfg.Sp, cfg.gamma);
    fprintf('Runs per case=%d, pseudo-Huber delta=%.3g\n', ...
        cfg.num_runs, cfg.huber_delta);
    if cfg.use_case_specific_budgets
        fprintf('Using case-specific stochastic-sample budgets.\n');
    else
        fprintf('Stochastic-sample budget per case=%.0f\n', cfg.sample_budget);
    end
    fprintf('============================================================\n');

    case_id = 0;
    for noise_index = 1:numel(cfg.noise_types)
        noise_type = cfg.noise_types{noise_index};

        for alpha_index = 1:numel(cfg.tail_indices)
            alpha = cfg.tail_indices(alpha_index);
            case_id = case_id + 1;

            case_cfg = cfg;
            case_cfg.sample_budget = get_case_budget( ...
                cfg, noise_type, alpha);
            checkpoints = make_record_checkpoints( ...
                case_cfg.sample_budget, case_cfg.num_record_points);

            fprintf('\n------------------------------------------------------------\n');
            fprintf('Case %d/%d: %s rewards, alpha=%.3g, budget=%.0f\n', ...
                case_id, numel(cfg.noise_types) * numel(cfg.tail_indices), ...
                upper(noise_type), alpha, case_cfg.sample_budget);
            fprintf('------------------------------------------------------------\n');

            raw = initialize_raw_results(case_cfg, numel(checkpoints));
            problem_statistics = zeros(cfg.num_runs, 4);

            for run = 1:cfg.num_runs
                reward_seed = make_reward_seed( ...
                    cfg, noise_type, alpha, run);
                common_algorithm_seed = make_algorithm_run_seed( ...
                    cfg, noise_type, alpha, run);
                problem = build_run_problem( ...
                    base_mdp, case_cfg, noise_type, alpha, reward_seed);

                problem_statistics(run, :) = [problem.w_star_norm, ...
                    min(problem.reward_table(:)), ...
                    median(problem.reward_table(:)), ...
                    max(problem.reward_table(:))];

                fprintf(['Run %2d/%2d: ||w*||=%.3e, ', ...
                    'reward range=[%.3e, %.3e], common seed=%.0f\n'], ...
                    run, cfg.num_runs, problem.w_star_norm, ...
                    problem_statistics(run, 2), problem_statistics(run, 4), ...
                    common_algorithm_seed);

                for algorithm_index = 1:numel(cfg.algorithm_order)
                    algorithm_name = cfg.algorithm_order{algorithm_index};

                    start_time = tic;
                    trajectory = run_algorithm(algorithm_name, problem, case_cfg, ...
                        checkpoints, common_algorithm_seed);
                    elapsed_time = toc(start_time);

                    raw.(algorithm_name).relative_error(run, :) = ...
                        trajectory.relative_error;
                    raw.(algorithm_name).relative_residual(run, :) = ...
                        trajectory.relative_residual;
                    raw.(algorithm_name).objective(run, :) = trajectory.objective;
                    raw.(algorithm_name).runtime(run) = elapsed_time;

                    fprintf('  %-12s final error=%9.3e, time=%8.2fs\n', ...
                        cfg.algorithm_labels.(algorithm_name), ...
                        trajectory.relative_error(end), elapsed_time);
                end
            end

            summary = aggregate_results(raw, case_cfg, checkpoints);
            base_name = make_case_name(noise_type, alpha);

            if cfg.save_mat
                save(fullfile(output_dir, [base_name, '.mat']), ...
                    'summary', 'raw', 'problem_statistics', 'case_cfg', ...
                    'noise_type', 'alpha', 'checkpoints', '-v7.3');
            end

            fig = plot_case_results(summary, case_cfg, noise_type, alpha);
            export_academic_figure(fig, fullfile(output_dir, base_name), case_cfg);

            fprintf('Mean final relative errors:\n');
            for algorithm_index = 1:numel(cfg.algorithm_order)
                algorithm_name = cfg.algorithm_order{algorithm_index};
                fprintf('  %-12s %.4e\n', ...
                    cfg.algorithm_labels.(algorithm_name), ...
                    summary.(algorithm_name).mean_relative_error(end));
            end
        end
    end

    fprintf('\nAll results were written to:\n  %s\n', output_dir);
end

%% ========================================================================
% Configuration
% =========================================================================
function cfg = default_config()


    %% Reproducibility and output
    cfg.problem_seed = 20260824;
    cfg.reward_seed = 31817;
    cfg.base_seed = 73129;

    cfg.num_runs = 25;

    cfg.use_case_specific_budgets = true;
    cfg.sample_budget = 1.5e7;  
    cfg.case_budget_tail_indices = [1.2, 1.5, 1.8];
    cfg.case_budgets.sas = [5e8, 5.0e7, 3e7];
    cfg.case_budgets.pareto = [1.0e9, 5.0e8, 3.0e8];
    cfg.num_record_points = 260;
    
    batch=50; ibatch=10;

    cfg.output_folder = 'results_minimal';
    cfg.save_mat = true;
    cfg.save_png = true;
    cfg.save_pdf = true;
    cfg.png_resolution = 600;
    cfg.divergence_error = 1.0e6;

    %% MDP
    cfg.S = 100;
    cfg.A = 3;
    cfg.Sp = 4;
    cfg.gamma = 0.90;

    %% heavy-tailed rewards
    cfg.noise_types = {'sas', 'pareto'};
    cfg.tail_indices = [1.2, 1.5, 1.8];

    % S-alpha-S characteristic function: exp(-c*|t|^alpha).
    cfg.sas_c = 5.0;

    % Pareto density: alpha*xm^alpha/x^(alpha+1), x >= xm.
    cfg.pareto_xm = 3.0;

    %% Outer objective
    % Mean pseudo-Huber loss. 
    cfg.huber_delta = 1.0;

    %% Algorithms
    % ASC-PG
    cfg.alg.ASC.init_batch = batch;
    cfg.alg.ASC.g_batch = batch;
    cfg.alg.ASC.J_batch = batch;
    cfg.alg.ASC.eta_x = 1.0e-2;
    cfg.alg.ASC.eta_y = 1.0e-2;
    cfg.alg.ASC.decay_start_fraction = 0.2;
    cfg.alg.ASC.decay_power = 1;

    % CIVR
    cfg.alg.CIVR.eta = 1.0e-2;
    cfg.alg.CIVR.tau = 4;
    cfg.alg.CIVR.outer_g_batch = batch;
    cfg.alg.CIVR.outer_J_batch = batch;
    cfg.alg.CIVR.inner_g_batch = ibatch;
    cfg.alg.CIVR.inner_J_batch = ibatch;
    cfg.alg.CIVR.project_J = false;
    cfg.alg.CIVR.J_radius = inf;

    % SCSC
    cfg.alg.SCSC.init_batch = batch;
    cfg.alg.SCSC.g_batch = batch;
    cfg.alg.SCSC.J_batch = batch;
    cfg.alg.SCSC.eta = 1.0e-2;
    cfg.alg.SCSC.beta = 1.0e-2;
    cfg.alg.SCSC.decay_start_fraction = 0.2;
    cfg.alg.SCSC.decay_power = 0.5;

    % MVRC 
    cfg.alg.MVRC.eta = cfg.alg.CIVR.eta;
    cfg.alg.MVRC.alpha = 0.8;
    cfg.alg.MVRC.beta = cfg.alg.MVRC.eta;
    cfg.alg.MVRC.lambda = 1.8 * cfg.alg.MVRC.eta;
    cfg.alg.MVRC.tau = cfg.alg.CIVR.tau;
    cfg.alg.MVRC.outer_g_batch = batch;
    cfg.alg.MVRC.outer_J_batch = batch;
    cfg.alg.MVRC.inner_g_batch = ibatch;
    cfg.alg.MVRC.inner_J_batch = ibatch;

    % Clipped-SCG
    cfg.alg.CLIP.eta = 1.0e-2;
    cfg.alg.CLIP.g_batch = batch;
    cfg.alg.CLIP.J_batch = batch;
    cfg.alg.CLIP.threshold = 1.0;

    % NSCG-M
    cfg.alg.NSCG_M.eta = 1.0e-2;
    cfg.alg.NSCG_M.g_batch = batch;
    cfg.alg.NSCG_M.J_batch = batch;

    % NSCG-S
    cfg.alg.NSCG_S.eta = 1.0e-2;
    cfg.alg.NSCG_S.tau = 4;
    cfg.alg.NSCG_S.outer_g_batch = batch;
    cfg.alg.NSCG_S.outer_J_batch = batch;
    cfg.alg.NSCG_S.inner_g_batch = ibatch;
    cfg.alg.NSCG_S.inner_J_batch = ibatch;
    cfg.alg.NSCG_S.project_J = false;
    cfg.alg.NSCG_S.J_radius_multiplier = 100;

    cfg.algorithm_order = {'ASC', 'CIVR', 'SCSC', 'MVRC', ...
        'CLIP', 'NSCG_M', 'NSCG_S'};

    cfg.algorithm_labels = struct( ...
        'ASC',        'ASC-PG', ...
        'CIVR',       'CIVR', ...
        'SCSC',       'SCSC', ...
        'MVRC',      'MVRC', ...
        'CLIP',       'Clipped-SCG', ...
        'NSCG_M',     'NSCG-M', ...
        'NSCG_S',     'NSCG-S');

    %% Plotting
    cfg.figure_position = [100, 100, 820, 600];
    cfg.font_size = 16;
    cfg.legend_font_size = 12;
    cfg.line_width = 2.5;
    cfg.marker_size = 7;
    cfg.min_positive_y = 1.0e-8;
    cfg.legend_location = 'best';

    cfg.plot_styles.ASC = struct('color', [0.35 0.35 0.35], ...
        'line', ':',  'marker', 'o');
    cfg.plot_styles.CIVR = struct('color', [0.90 0.60 0.00], ...
        'line', '--', 'marker', 's');
    cfg.plot_styles.SCSC = struct('color', [0.00 0.45 0.70], ...
        'line', '-.', 'marker', '^');
    cfg.plot_styles.MVRC = struct('color', [0.49 0.18 0.56], ...
        'line', '--', 'marker', 'd');
    cfg.plot_styles.CLIP = struct('color', [0.80 0.40 0.00], ...
        'line', '-',  'marker', 'v');
    cfg.plot_styles.NSCG_M = struct('color', [0.00 0.62 0.45], ...
        'line', '--', 'marker', '>');
    cfg.plot_styles.NSCG_S = struct('color', [0.84 0.15 0.16], ...
        'line', '-',  'marker', 'p');
end

function validate_config(cfg)
    assert(cfg.S >= 2 && cfg.A >= 1 && cfg.Sp >= 1, ...
        'Invalid MDP dimensions.');
    assert(cfg.Sp <= cfg.S, 'Sp must not exceed S.');
    assert(cfg.gamma > 0 && cfg.gamma < 1, ...
        'gamma must lie in (0,1).');
    assert(cfg.num_runs >= 1 && cfg.sample_budget >= 1, ...
        'num_runs and sample_budget must be positive.');
    assert(isnumeric(cfg.tail_indices) && isvector(cfg.tail_indices) && ...
        ~isempty(cfg.tail_indices), ...
        'cfg.tail_indices must be a nonempty numeric scalar or vector.');
    assert(all(isfinite(cfg.tail_indices)) && ...
        all(cfg.tail_indices > 1) && all(cfg.tail_indices <= 2), ...
        'Every tail index must lie in (1,2].');
    assert(isnumeric(cfg.case_budget_tail_indices) && ...
        isvector(cfg.case_budget_tail_indices) && ...
        numel(unique(cfg.case_budget_tail_indices)) == ...
        numel(cfg.case_budget_tail_indices), ...
        'cfg.case_budget_tail_indices must contain distinct values.');
    assert(numel(cfg.case_budgets.sas) == 1 || ...
        numel(cfg.case_budgets.sas) == ...
        numel(cfg.case_budget_tail_indices), ...
        ['cfg.case_budgets.sas must be scalar or match ', ...
         'cfg.case_budget_tail_indices.']);
    assert(numel(cfg.case_budgets.pareto) == 1 || ...
        numel(cfg.case_budgets.pareto) == ...
        numel(cfg.case_budget_tail_indices), ...
        ['cfg.case_budgets.pareto must be scalar or match ', ...
         'cfg.case_budget_tail_indices.']);
    assert(all(cfg.case_budgets.sas > 0) && ...
        all(cfg.case_budgets.pareto > 0), ...
        'Every stochastic-sample budget must be positive.');
    assert(isscalar(cfg.huber_delta) && isfinite(cfg.huber_delta) && ...
        cfg.huber_delta > 0, ...
        'cfg.huber_delta must be a positive finite scalar.');
    assert(cfg.alg.ASC.eta_x > 0 && cfg.alg.ASC.eta_y > 0 && ...
        cfg.alg.ASC.eta_y <= 1, ...
        'ASC-PG requires eta_x>0 and eta_y in (0,1].');
    assert(is_positive_integer(cfg.alg.ASC.init_batch) && ...
        is_positive_integer(cfg.alg.ASC.g_batch) && ...
        is_positive_integer(cfg.alg.ASC.J_batch), ...
        'ASC-PG batch sizes must be positive integers.');
    assert(cfg.alg.ASC.decay_start_fraction > 0 && ...
        cfg.alg.ASC.decay_start_fraction <= 1 && ...
        cfg.alg.ASC.decay_power > 0, ...
        ['ASC-PG decay_start_fraction must lie in (0,1] and ', ...
         'decay_power must be positive.']);
    assert(cfg.alg.SCSC.beta > 0 && cfg.alg.SCSC.beta <= 1, ...
        'SCSC beta must lie in (0,1].');
    assert(is_positive_integer(cfg.alg.SCSC.init_batch) && ...
        is_positive_integer(cfg.alg.SCSC.g_batch) && ...
        is_positive_integer(cfg.alg.SCSC.J_batch), ...
        'SCSC batch sizes must be positive integers.');
    assert(cfg.alg.SCSC.decay_start_fraction > 0 && ...
        cfg.alg.SCSC.decay_start_fraction <= 1 && ...
        cfg.alg.SCSC.decay_power > 0, ...
        ['SCSC decay_start_fraction must lie in (0,1] and ', ...
         'decay_power must be positive.']);
    assert(cfg.alg.MVRC.alpha > 0 && ...
        cfg.alg.MVRC.alpha <= 1, ...
        'MVRC alpha must lie in (0,1].');
    assert(cfg.alg.MVRC.beta > 0 && cfg.alg.MVRC.lambda > 0, ...
        'MVRC beta and lambda must be positive.');
    mvrc_tolerance = 100 * eps(max(1, cfg.alg.MVRC.lambda));
    assert(cfg.alg.MVRC.lambda + mvrc_tolerance >= ...
        cfg.alg.MVRC.beta && ...
        cfg.alg.MVRC.lambda <= ...
        (1 + cfg.alg.MVRC.alpha) * cfg.alg.MVRC.beta + ...
        mvrc_tolerance, ...
        ['MVRC requires beta <= lambda <= ', ...
         '(1+alpha)*beta.']);
    assert(is_positive_integer(cfg.alg.MVRC.tau) && ...
        is_positive_integer(cfg.alg.MVRC.outer_g_batch) && ...
        is_positive_integer(cfg.alg.MVRC.outer_J_batch) && ...
        is_positive_integer(cfg.alg.MVRC.inner_g_batch) && ...
        is_positive_integer(cfg.alg.MVRC.inner_J_batch), ...
        'MVRC tau and all batch sizes must be positive integers.');
    assert(cfg.alg.MVRC.tau == cfg.alg.CIVR.tau && ...
        cfg.alg.MVRC.outer_g_batch == cfg.alg.CIVR.outer_g_batch && ...
        cfg.alg.MVRC.outer_J_batch == cfg.alg.CIVR.outer_J_batch && ...
        cfg.alg.MVRC.inner_g_batch == cfg.alg.CIVR.inner_g_batch && ...
        cfg.alg.MVRC.inner_J_batch == cfg.alg.CIVR.inner_J_batch, ...
        ['MVRC must use the same tau and G/J batch sizes as CIVR ', ...
         'for the requested sample-cost comparison.']);
    assert(~cfg.alg.CIVR.project_J && ~cfg.alg.NSCG_S.project_J, ...
        ['This minimal implementation intentionally omits SVD-based ', ...
         'Jacobian projection. Set both project_J flags to false.']);
end

%% ========================================================================
% MDP and heavy-tailed reward generation
% =========================================================================
function base = build_base_mdp(cfg)

    rng(cfg.problem_seed, 'twister');

    P = rand(cfg.S, cfg.A, cfg.Sp);
    P = P ./ sum(P, 3);

    successor_map = zeros(cfg.S, cfg.A, cfg.Sp, 'uint16');
    for state = 1:cfg.S
        for action = 1:cfg.A
            successor_map(state, action, :) = ...
                uint16(randperm(cfg.S, cfg.Sp));
        end
    end

    policy = ones(cfg.S, cfg.A) / cfg.A;
    P_pi = zeros(cfg.S, cfg.S);

    for state = 1:cfg.S
        for action = 1:cfg.A
            for next_index = 1:cfg.Sp
                next_state = double(successor_map(state, action, next_index));
                probability = policy(state, action) * ...
                    P(state, action, next_index);
                P_pi(state, next_state) = ...
                    P_pi(state, next_state) + probability;
            end
        end
    end

    P_pi = P_pi ./ sum(P_pi, 2);

    base = struct();
    base.P = P;
    base.successor_map = successor_map;
    base.policy = policy;
    base.P_pi = P_pi;
    base.Phi = eye(cfg.S);
end

function problem = build_run_problem(base, cfg, noise_type, alpha, reward_seed)

    rng(reward_seed, 'twister');
    reward_table = draw_reward_table(cfg, noise_type, alpha);

    r_pi = zeros(cfg.S, 1);
    for state = 1:cfg.S
        for action = 1:cfg.A
            for next_index = 1:cfg.Sp
                probability = base.policy(state, action) * ...
                    base.P(state, action, next_index);
                r_pi(state) = r_pi(state) + probability * ...
                    reward_table(state, action, next_index);
            end
        end
    end

    value_map = base.Phi;
    continuation_map = cfg.gamma * base.P_pi * base.Phi;
    bellman_map = value_map - continuation_map;
    w_star = bellman_map \ r_pi;

    problem = struct();
    problem.S = cfg.S;
    problem.d = size(base.Phi, 2);
    problem.gamma = cfg.gamma;
    problem.P_pi = base.P_pi;
    problem.Phi = base.Phi;
    problem.value_map = value_map;
    problem.continuation_map = continuation_map;
    problem.bellman_map = bellman_map;
    problem.r_pi = r_pi;
    problem.w_star = w_star;
    problem.w_star_norm = max(norm(w_star), eps);
    problem.r_pi_norm = max(norm(r_pi), eps);
    problem.reward_table = reward_table;
    problem.huber_delta = cfg.huber_delta;
    problem.divergence_error = cfg.divergence_error;
end

function reward_table = draw_reward_table(cfg, noise_type, alpha)
    number_of_rewards = cfg.S * cfg.A * cfg.Sp;

    switch lower(noise_type)
        case 'sas'
            values = draw_symmetric_alpha_stable( ...
                number_of_rewards, alpha, cfg.sas_c);

        case 'pareto'
            values = draw_pareto( ...
                number_of_rewards, alpha, cfg.pareto_xm);

        otherwise
            error('Unknown reward distribution "%s".', noise_type);
    end

    reward_table = reshape(values, cfg.S, cfg.A, cfg.Sp);
end

function values = draw_symmetric_alpha_stable(n, alpha, c)

    assert(alpha > 1 && alpha <= 2, ...
        'The S-alpha-S tail index must lie in (1,2].');
    assert(c > 0, 'The S-alpha-S scale c must be positive.');

    if alpha == 2
        values = sqrt(2 * c) .* randn(n, 1);
        return;
    end

    values = zeros(n, 1);
    unfinished = true(n, 1);
    sigma = c^(1 / alpha);

    while any(unfinished)
        m = sum(unfinished);
        U = pi * (rand(m, 1) - 0.5);
        W = -log(max(rand(m, 1), realmin('double')));

        standard_values = sin(alpha .* U) ./ ...
            (cos(U).^(1 / alpha)) .* ...
            (cos((1 - alpha) .* U) ./ W).^((1 - alpha) / alpha);
        candidate = sigma .* standard_values;

        finite_mask = isfinite(candidate);
        unfinished_indices = find(unfinished);
        values(unfinished_indices(finite_mask)) = candidate(finite_mask);
        unfinished(unfinished_indices(finite_mask)) = false;
    end
end

function values = draw_pareto(n, alpha, x_m)

    assert(alpha > 1 && alpha <= 2, ...
        'The Pareto tail index must lie in (1,2].');
    assert(x_m > 0, 'The Pareto threshold x_m must be positive.');

    u = max(rand(n, 1), realmin('double'));
    values = x_m .* u.^(-1 / alpha);

    nonfinite = ~isfinite(values);
    while any(nonfinite)
        u = max(rand(sum(nonfinite), 1), realmin('double'));
        values(nonfinite) = x_m .* u.^(-1 / alpha);
        nonfinite = ~isfinite(values);
    end
end

%% ========================================================================
% Experiment bookkeeping
% =========================================================================
function raw = initialize_raw_results(cfg, number_of_points)
    raw = struct();
    for k = 1:numel(cfg.algorithm_order)
        name = cfg.algorithm_order{k};
        raw.(name).relative_error = zeros(cfg.num_runs, number_of_points);
        raw.(name).relative_residual = zeros(cfg.num_runs, number_of_points);
        raw.(name).objective = zeros(cfg.num_runs, number_of_points);
        raw.(name).runtime = zeros(cfg.num_runs, 1);
    end
end

function summary = aggregate_results(raw, cfg, checkpoints)
    summary = struct();
    z_value = 1.959963984540054;

    for k = 1:numel(cfg.algorithm_order)
        name = cfg.algorithm_order{k};
        errors = raw.(name).relative_error;
        residuals = raw.(name).relative_residual;
        objectives = raw.(name).objective;

        mean_error = mean(errors, 1);
        mean_residual = mean(residuals, 1);
        mean_objective = mean(objectives, 1);

        if cfg.num_runs > 1
            standard_error = std(errors, 0, 1) / sqrt(cfg.num_runs);
        else
            standard_error = zeros(size(mean_error));
        end

        summary.(name).samples = checkpoints;
        summary.(name).mean_relative_error = mean_error;
        summary.(name).ci_lower = max( ...
            mean_error - z_value .* standard_error, cfg.min_positive_y);
        summary.(name).ci_upper = mean_error + z_value .* standard_error;
        summary.(name).mean_relative_residual = mean_residual;
        summary.(name).mean_objective = mean_objective;
        summary.(name).mean_runtime = mean(raw.(name).runtime);
    end
end

function seed = make_reward_seed(cfg, noise_type, alpha, run)

    case_code = make_case_seed_code(noise_type, alpha);
    seed = cfg.reward_seed + 300000000 + 1000 * case_code + run;
end

function seed = make_algorithm_run_seed(cfg, noise_type, alpha, run)

    case_code = make_case_seed_code(noise_type, alpha);
    seed = cfg.base_seed + 1000 * case_code + run;
end

function code = make_case_seed_code(noise_type, alpha)
    switch lower(noise_type)
        case 'sas'
            noise_code = 1;
        case 'pareto'
            noise_code = 2;
        otherwise
            error('Unknown reward distribution "%s".', noise_type);
    end

    alpha_code = round(1000 * alpha);
    code = 100000 * noise_code + alpha_code;
end

function budget = get_case_budget(cfg, noise_type, alpha)

    if ~cfg.use_case_specific_budgets
        budget = cfg.sample_budget;
        return;
    end

    switch lower(noise_type)
        case 'sas'
            budget_table = cfg.case_budgets.sas;
        case 'pareto'
            budget_table = cfg.case_budgets.pareto;
        otherwise
            error('Unknown reward distribution "%s".', noise_type);
    end

    if isscalar(budget_table)
        budget = budget_table;
        return;
    end

    tolerance = 100 * eps(max(1, abs(alpha)));
    match = find(abs(cfg.case_budget_tail_indices - alpha) <= ...
        tolerance, 1, 'first');

    if isempty(match)
        budget = cfg.sample_budget;
        warning('main_aver:BudgetFallback', ...
            ['No case-specific budget is tabulated for %s alpha=%.6g. ', ...
             'Using cfg.sample_budget=%.0f.'], ...
            upper(noise_type), alpha, budget);
    else
        budget = budget_table(match);
    end
end

function checkpoints = make_record_checkpoints(sample_budget, requested_points)
    requested_points = max(20, round(requested_points));
    early_points = max(10, round(0.35 * requested_points));
    late_points = max(10, requested_points - early_points);
    split_budget = max(10, round(0.08 * sample_budget));

    early = round(logspace(0, log10(split_budget), early_points));
    late = round(linspace(split_budget, sample_budget, late_points));
    checkpoints = unique([0, early, late, sample_budget]);
end

function name = make_case_name(noise_type, alpha)
    alpha_tag = strrep(sprintf('%.6g', alpha), '.', 'p');
    name = sprintf('convergence_%s_alpha_%s', lower(noise_type), alpha_tag);
end

%% ========================================================================
% Algorithm dispatcher
% =========================================================================
function trajectory = run_algorithm(name, problem, cfg, checkpoints, seed)
    switch upper(name)
        case 'ASC'
            trajectory = run_asc_pg(problem, cfg.alg.ASC, ...
                cfg.sample_budget, checkpoints, seed);
        case 'CIVR'
            trajectory = run_civr(problem, cfg.alg.CIVR, ...
                cfg.sample_budget, checkpoints, seed);
        case 'SCSC'
            trajectory = run_scsc(problem, cfg.alg.SCSC, ...
                cfg.sample_budget, checkpoints, seed);
        case 'MVRC'
            trajectory = run_MVRC(problem, cfg.alg.MVRC, ...
                cfg.sample_budget, checkpoints, seed);
        case 'CLIP'
            trajectory = run_clipped_scg(problem, cfg.alg.CLIP, ...
                cfg.sample_budget, checkpoints, seed);
        case 'NSCG_M'
            trajectory = run_nscg_m(problem, cfg.alg.NSCG_M, ...
                cfg.sample_budget, checkpoints, seed);
        case 'NSCG_S'
            trajectory = run_nscg_s(problem, cfg.alg.NSCG_S, ...
                cfg.sample_budget, checkpoints, seed);
        otherwise
            error('Unknown algorithm "%s".', name);
    end
end

%% ========================================================================
% ASC-PG
% =========================================================================
function trajectory = run_asc_pg(problem, alg, budget, checkpoints, seed)

    rng(seed, 'twister');

    w = zeros(problem.d, 1);
    z = w;
    samples_used = 0;

    trajectory = initialize_trajectory(checkpoints, w, problem);

    if alg.init_batch > budget
        trajectory = finalize_trajectory(trajectory, w, problem);
        return;
    end

    y = estimate_g_batch(problem, z, alg.init_batch);
    samples_used = samples_used + alg.init_batch;
    trajectory = record_if_reached( ...
        trajectory, samples_used, w, problem);

    step_cost = alg.J_batch + alg.g_batch;
    max_iterations = floor((budget - samples_used) / step_cost);
    iteration = 0;

    while samples_used + step_cost <= budget
        iteration = iteration + 1;

        decay_factor = constant_then_power_decay( ...
            iteration, max_iterations, ...
            alg.decay_start_fraction, alg.decay_power);

        alpha_k = alg.eta_x * decay_factor;
        beta_k  = alg.eta_y * decay_factor;

        outer_gradient = pseudo_huber_outer_gradient(y, problem);

        gradient_estimate = estimate_Jt_batch( ...
            problem, outer_gradient, alg.J_batch);
        w_new = w - alpha_k .* gradient_estimate;

        z_new = w - (alpha_k / beta_k) .* gradient_estimate;

        g_new = estimate_g_batch(problem, z_new, alg.g_batch);
        y_new = (1 - beta_k) .* y + beta_k .* g_new;

        w = w_new;
        z = z_new;
        y = y_new;
        samples_used = samples_used + step_cost;

        if trajectory.next_index <= numel(checkpoints) && ...
                samples_used >= checkpoints(trajectory.next_index)
            trajectory = record_if_reached( ...
                trajectory, samples_used, w, problem);
        end

        if has_diverged(w)
            w(:) = NaN;
            break;
        end
    end

    trajectory = finalize_trajectory(trajectory, w, problem);
end

%% ========================================================================
% CIVR
% =========================================================================
function trajectory = run_civr(problem, alg, budget, checkpoints, seed)
    rng(seed, 'twister');

    w = zeros(problem.d, 1);
    samples_used = 0;
    trajectory = initialize_trajectory(checkpoints, w, problem);

    outer_cost = alg.outer_g_batch + alg.outer_J_batch;
    inner_cost = alg.inner_g_batch + alg.inner_J_batch;

    while samples_used + outer_cost <= budget
        w_outer = w;
        g_estimate = estimate_g_batch( ...
            problem, w_outer, alg.outer_g_batch);
        J_weights = estimate_J_weights(problem, alg.outer_J_batch);
        samples_used = samples_used + outer_cost;

        gradient_estimate = apply_J_weights( ...
            problem, J_weights, pseudo_huber_outer_gradient(g_estimate, problem));
        w_current = w_outer - alg.eta .* gradient_estimate;

        trajectory = record_if_reached( ...
            trajectory, samples_used, w_current, problem);
        if has_diverged(w_current)
            w = NaN(problem.d, 1);
            break;
        end

        w_previous = w_outer;
        for inner_index = 2:alg.tau 
            if samples_used + inner_cost > budget
                break;
            end

            g_estimate = g_estimate + estimate_g_difference( ...
                problem, w_current, w_previous, alg.inner_g_batch);
            consume_J_difference_samples(problem, alg.inner_J_batch);
            samples_used = samples_used + inner_cost;

            gradient_estimate = apply_J_weights( ...
                problem, J_weights, pseudo_huber_outer_gradient(g_estimate, problem));
            w_next = w_current - alg.eta .* gradient_estimate;

            w_previous = w_current;
            w_current = w_next;

            if trajectory.next_index <= numel(checkpoints) && ...
                    samples_used >= checkpoints(trajectory.next_index)
                trajectory = record_if_reached( ...
                    trajectory, samples_used, w_current, problem);
            end

            if has_diverged(w_current)
                w_current(:) = NaN;
                break;
            end
        end

        w = w_current;
        if has_diverged(w)
            w(:) = NaN;
            break;
        end
    end

    trajectory = finalize_trajectory(trajectory, w, problem);
end

%% ========================================================================
% SCSC
% =========================================================================
function trajectory = run_scsc(problem, alg, budget, checkpoints, seed)

    rng(seed, 'twister');

    w = zeros(problem.d, 1);
    w_previous = w;
    samples_used = 0;
    trajectory = initialize_trajectory(checkpoints, w, problem);

    if alg.init_batch > budget
        trajectory = finalize_trajectory(trajectory, w, problem);
        return;
    end

    g_previous = estimate_g_batch(problem, w, alg.init_batch);
    samples_used = samples_used + alg.init_batch;
    trajectory = record_if_reached( ...
        trajectory, samples_used, w, problem);

    step_cost = alg.g_batch + alg.J_batch;
    max_iterations = floor((budget - samples_used) / step_cost);
    iteration = 0;

    while samples_used + step_cost <= budget
        iteration = iteration + 1;

        decay_factor = constant_then_power_decay( ...
            iteration, max_iterations, ...
            alg.decay_start_fraction, alg.decay_power);
        eta_k = alg.eta * decay_factor;
        beta_k = alg.beta * decay_factor;

        [g_current_batch, g_previous_batch] = ...
            estimate_paired_g_batches( ...
                problem, w, w_previous, alg.g_batch);

        g_corrected = (1 - beta_k) .* g_previous + ...
            beta_k .* g_current_batch + ...
            (1 - beta_k) .* ...
            (g_current_batch - g_previous_batch);

        outer_gradient = pseudo_huber_outer_gradient(g_corrected, problem);

        gradient_estimate = estimate_Jt_batch( ...
            problem, outer_gradient, alg.J_batch);
        w_new = w - eta_k .* gradient_estimate;

        w_previous = w;
        w = w_new;
        g_previous = g_corrected;
        samples_used = samples_used + step_cost;

        if trajectory.next_index <= numel(checkpoints) && ...
                samples_used >= checkpoints(trajectory.next_index)
            trajectory = record_if_reached( ...
                trajectory, samples_used, w, problem);
        end

        if has_diverged(w)
            w(:) = NaN;
            break;
        end
    end

    trajectory = finalize_trajectory(trajectory, w, problem);
end

%% ========================================================================
% MVRC
% =========================================================================
function trajectory = run_MVRC(problem, alg, budget, checkpoints, seed)

    rng(seed, 'twister');

    x = zeros(problem.d, 1);
    y = x;
    reported_point = x;
    samples_used = 0;
    iteration = 0;

    trajectory = initialize_trajectory(checkpoints, reported_point, problem);

    g_estimate = [];
    J_weights = [];
    z_previous = [];

    outer_cost = alg.outer_g_batch + alg.outer_J_batch;
    inner_cost = alg.inner_g_batch + alg.inner_J_batch;

    while true
        refresh = mod(iteration, alg.tau) == 0;
        if refresh
            step_cost = outer_cost;
        else
            step_cost = inner_cost;
        end

        if samples_used + step_cost > budget
            break;
        end

        z = (1 - alg.alpha) .* y + alg.alpha .* x;

        if refresh
            g_estimate = estimate_MVRC_g_batch( ...
                problem, z, alg.outer_g_batch);
            J_weights = estimate_MVRC_J_weights( ...
                problem, alg.outer_J_batch);
        else
            g_estimate = g_estimate + estimate_MVRC_g_difference( ...
                problem, z, z_previous, alg.inner_g_batch);
            J_weights = J_weights + estimate_MVRC_J_weight_difference( ...
                problem, alg.inner_J_batch);
        end
        samples_used = samples_used + step_cost;

        outer_gradient = pseudo_huber_outer_gradient(g_estimate, problem);
        gradient_estimate = apply_J_weights( ...
            problem, J_weights, outer_gradient);

        x_next = x - alg.lambda .* gradient_estimate;
        y_next = z + (alg.beta / alg.lambda) .* (x_next - x);

        state_has_diverged = has_diverged(x_next) || ...
            has_diverged(y_next) || has_diverged(z);
        if state_has_diverged
            reported_point = NaN(problem.d, 1);
        else
            reported_point = z;
        end

        trajectory = record_if_reached( ...
            trajectory, samples_used, reported_point, problem);

        if state_has_diverged
            break;
        end

        x = x_next;
        y = y_next;
        z_previous = z;
        iteration = iteration + 1;
    end

    trajectory = finalize_trajectory( ...
        trajectory, reported_point, problem);
end

%% ========================================================================
% Clipped-SCG
% =========================================================================
function trajectory = run_clipped_scg(problem, alg, budget, checkpoints, seed)
    rng(seed, 'twister');

    w = zeros(problem.d, 1);
    samples_used = 0;
    trajectory = initialize_trajectory(checkpoints, w, problem);

    step_cost = alg.g_batch + alg.J_batch;
    while samples_used + step_cost <= budget
        g_estimate = estimate_g_batch(problem, w, alg.g_batch);
        outer_gradient = pseudo_huber_outer_gradient(g_estimate, problem);
        gradient_estimate = estimate_Jt_batch( ...
            problem, outer_gradient, alg.J_batch);

        w = w - alg.eta .* clip_vector( ...
            gradient_estimate, alg.threshold);
        samples_used = samples_used + step_cost;

        if trajectory.next_index <= numel(checkpoints) && ...
                samples_used >= checkpoints(trajectory.next_index)
            trajectory = record_if_reached( ...
                trajectory, samples_used, w, problem);
        end

        if has_diverged(w)
            w(:) = NaN;
            break;
        end
    end

    trajectory = finalize_trajectory(trajectory, w, problem);
end

%% ========================================================================
% NSCG-M
% =========================================================================
function trajectory = run_nscg_m(problem, alg, budget, checkpoints, seed)
    rng(seed, 'twister');

    w = zeros(problem.d, 1);
    samples_used = 0;
    trajectory = initialize_trajectory(checkpoints, w, problem);

    step_cost = alg.g_batch + alg.J_batch;
    while samples_used + step_cost <= budget
        g_estimate = estimate_g_batch(problem, w, alg.g_batch);
        outer_gradient = pseudo_huber_outer_gradient(g_estimate, problem);
        gradient_estimate = estimate_Jt_batch( ...
            problem, outer_gradient, alg.J_batch);

        w = w - alg.eta .* safe_normalize(gradient_estimate);
        samples_used = samples_used + step_cost;

        if trajectory.next_index <= numel(checkpoints) && ...
                samples_used >= checkpoints(trajectory.next_index)
            trajectory = record_if_reached( ...
                trajectory, samples_used, w, problem);
        end

        if has_diverged(w)
            w(:) = NaN;
            break;
        end
    end

    trajectory = finalize_trajectory(trajectory, w, problem);
end

%% ========================================================================
% NSCG-S
% =========================================================================
function trajectory = run_nscg_s(problem, alg, budget, checkpoints, seed)
    rng(seed, 'twister');

    w = zeros(problem.d, 1);
    samples_used = 0;
    trajectory = initialize_trajectory(checkpoints, w, problem);

    outer_cost = alg.outer_g_batch + alg.outer_J_batch;
    inner_cost = alg.inner_g_batch + alg.inner_J_batch;

    while samples_used + outer_cost <= budget
        w_outer = w;
        g_estimate = estimate_g_batch( ...
            problem, w_outer, alg.outer_g_batch);
        J_weights = estimate_J_weights(problem, alg.outer_J_batch);
        samples_used = samples_used + outer_cost;

        gradient_estimate = apply_J_weights( ...
            problem, J_weights, pseudo_huber_outer_gradient(g_estimate, problem));
        w_current = w_outer - alg.eta .* safe_normalize(gradient_estimate);

        trajectory = record_if_reached( ...
            trajectory, samples_used, w_current, problem);
        if has_diverged(w_current)
            w = NaN(problem.d, 1);
            break;
        end

        w_previous = w_outer;
        for inner_index = 2:alg.tau 
            if samples_used + inner_cost > budget
                break;
            end

            g_estimate = g_estimate + estimate_g_difference( ...
                problem, w_current, w_previous, alg.inner_g_batch);
            consume_J_difference_samples(problem, alg.inner_J_batch);
            samples_used = samples_used + inner_cost;

            gradient_estimate = apply_J_weights( ...
                problem, J_weights, pseudo_huber_outer_gradient(g_estimate, problem));
            w_next = w_current - alg.eta .* ...
                safe_normalize(gradient_estimate);

            w_previous = w_current;
            w_current = w_next;

            if trajectory.next_index <= numel(checkpoints) && ...
                    samples_used >= checkpoints(trajectory.next_index)
                trajectory = record_if_reached( ...
                    trajectory, samples_used, w_current, problem);
            end

            if has_diverged(w_current)
                w_current(:) = NaN;
                break;
            end
        end

        w = w_current;
        if has_diverged(w)
            w(:) = NaN;
            break;
        end
    end

    trajectory = finalize_trajectory(trajectory, w, problem);
end

%% ========================================================================
% Stochastic compositional oracles
% =========================================================================
function g_estimate = estimate_MVRC_g_batch(problem, z, batch_size)

    states = randi(problem.S, batch_size, 1);
    counts = accumarray(states, 1, [problem.S, 1], @sum, 0);
    scale = problem.S / batch_size;

    value = problem.value_map * z;
    q_value = problem.r_pi + problem.continuation_map * z;

    g_estimate = zeros(2 * problem.S, 1);
    g_estimate(1:2:end) = scale .* counts .* value;
    g_estimate(2:2:end) = scale .* counts .* q_value;
end

function J_weights = estimate_MVRC_J_weights(problem, batch_size)

    states = randi(problem.S, batch_size, 1);
    counts = accumarray(states, 1, [problem.S, 1], @sum, 0);
    J_weights = (problem.S / batch_size) .* counts;
end

function g_difference = estimate_MVRC_g_difference( ...
        problem, z_new, z_old, batch_size)

    states = randi(problem.S, batch_size, 1);
    counts = accumarray(states, 1, [problem.S, 1], @sum, 0);
    scale = problem.S / batch_size;
    displacement = z_new - z_old;

    first_difference = problem.value_map * displacement;
    second_difference = problem.continuation_map * displacement;

    g_difference = zeros(2 * problem.S, 1);
    g_difference(1:2:end) = ...
        scale .* counts .* first_difference;
    g_difference(2:2:end) = ...
        scale .* counts .* second_difference;
end

function J_weight_difference = ...
        estimate_MVRC_J_weight_difference(problem, batch_size)

    randi(problem.S, batch_size, 1);
    J_weight_difference = zeros(problem.S, 1);
end

function g_estimate = estimate_g_batch(problem, w, batch_size)

    states = randi(problem.S, batch_size, 1);
    counts = accumarray(states, 1, [problem.S, 1], @sum, 0);

    value = problem.value_map * w;
    q_value = problem.r_pi + problem.continuation_map * w;
    scale = problem.S / batch_size;

    g_estimate = zeros(2 * problem.S, 1);
    g_estimate(1:2:end) = scale .* counts .* value;
    g_estimate(2:2:end) = scale .* counts .* q_value;
end

function [g_new, g_old] = estimate_paired_g_batches( ...
        problem, w_new, w_old, batch_size)

    states = randi(problem.S, batch_size, 1);
    counts = accumarray(states, 1, [problem.S, 1], @sum, 0);
    scale = problem.S / batch_size;

    value_new = problem.value_map * w_new;
    q_new = problem.r_pi + problem.continuation_map * w_new;
    value_old = problem.value_map * w_old;
    q_old = problem.r_pi + problem.continuation_map * w_old;

    g_new = zeros(2 * problem.S, 1);
    g_new(1:2:end) = scale .* counts .* value_new;
    g_new(2:2:end) = scale .* counts .* q_new;

    g_old = zeros(2 * problem.S, 1);
    g_old(1:2:end) = scale .* counts .* value_old;
    g_old(2:2:end) = scale .* counts .* q_old;
end

function difference = estimate_g_difference( ...
        problem, w_new, w_old, batch_size)

    states = randi(problem.S, batch_size, 1);
    counts = accumarray(states, 1, [problem.S, 1], @sum, 0);
    displacement = w_new - w_old;

    first_difference = problem.value_map * displacement;
    second_difference = problem.continuation_map * displacement;
    scale = problem.S / batch_size;

    difference = zeros(2 * problem.S, 1);
    difference(1:2:end) = scale .* counts .* first_difference;
    difference(2:2:end) = scale .* counts .* second_difference;
end

function weights = estimate_J_weights(problem, batch_size)

    states = randi(problem.S, batch_size, 1);
    counts = accumarray(states, 1, [problem.S, 1], @sum, 0);
    weights = (problem.S / batch_size) .* counts;
end

function product = estimate_Jt_batch(problem, vector, batch_size)
    weights = estimate_J_weights(problem, batch_size);
    product = apply_J_weights(problem, weights, vector);
end

function product = apply_J_weights(problem, state_weights, vector)
    odd_part = vector(1:2:end);
    even_part = vector(2:2:end);

    product = problem.value_map' * (state_weights .* odd_part) + ...
        problem.continuation_map' * (state_weights .* even_part);
end

function consume_J_difference_samples(problem, batch_size)

    randi(problem.S, batch_size, 1); 
end

function gradient = pseudo_huber_outer_gradient(y, problem)

    residual = y(1:2:end) - y(2:2:end);
    psi = pseudo_huber_derivative(residual, problem.huber_delta);

    gradient = zeros(size(y));
    gradient(1:2:end) = psi / problem.S;
    gradient(2:2:end) = -psi / problem.S;
end

function psi = pseudo_huber_derivative(residual, delta)

    psi = delta .* residual ./ hypot(delta, residual);
    infinite_mask = isinf(residual);
    psi(infinite_mask) = delta .* sign(residual(infinite_mask));
end

function values = pseudo_huber_loss(residual, delta)


    values = delta .* (hypot(delta, residual) - delta);
end


function flag = is_positive_integer(value)

    flag = isscalar(value) && isnumeric(value) && isfinite(value) && ...
        value >= 1 && value == floor(value);
end

function factor = constant_then_power_decay( ...
        iteration, total_iterations, start_fraction, power)

    if total_iterations <= 0
        factor = 1;
        return;
    end

    start_iteration = max(1, ...
        floor(start_fraction * total_iterations));

    if iteration <= start_iteration
        factor = 1;
    else
        factor = (start_iteration / iteration) ^ power;
    end
end

%% ========================================================================
% Numerical safeguards and trajectory recording
% =========================================================================
function direction = safe_normalize(vector)

    if isempty(vector)
        direction = vector;
        return;
    end

    infinite_mask = isinf(vector);
    if any(infinite_mask)
        direction = zeros(size(vector));
        signs = sign(vector(infinite_mask));
        direction(infinite_mask) = signs ./ sqrt(sum(infinite_mask));
        return;
    end

    scale = max(abs(vector));
    if scale == 0 || ~isfinite(scale)
        direction = zeros(size(vector));
        return;
    end

    scaled_vector = vector ./ scale;
    scaled_norm = norm(scaled_vector);
    if scaled_norm == 0 || ~isfinite(scaled_norm)
        direction = zeros(size(vector));
    else
        direction = scaled_vector ./ scaled_norm;
    end
end

function clipped = clip_vector(vector, threshold)
    assert(threshold > 0, 'The clipping threshold must be positive.');

    vector_norm = stable_norm(vector);
    if vector_norm <= threshold
        clipped = vector;
    elseif isfinite(vector_norm)
        clipped = (threshold / vector_norm) .* vector;
    else
        clipped = threshold .* safe_normalize(vector);
    end
end

function value = stable_norm(vector)
    infinite_mask = isinf(vector);
    if any(infinite_mask)
        value = inf;
        return;
    end

    scale = max(abs(vector));
    if scale == 0
        value = 0;
    elseif ~isfinite(scale)
        value = inf;
    else
        value = scale * norm(vector ./ scale);
    end
end

function flag = has_diverged(w)
    flag = any(~isfinite(w)) || max(abs(w)) > 1.0e100;
end

function trajectory = initialize_trajectory(checkpoints, w, problem)
    number_of_points = numel(checkpoints);
    trajectory.samples = checkpoints;
    trajectory.relative_error = NaN(1, number_of_points);
    trajectory.relative_residual = NaN(1, number_of_points);
    trajectory.objective = NaN(1, number_of_points);
    trajectory.next_index = 1;

    trajectory = record_if_reached(trajectory, 0, w, problem);
end

function trajectory = record_if_reached( ...
        trajectory, samples_used, w, problem)

    if trajectory.next_index > numel(trajectory.samples)
        return;
    end

    metrics = evaluate_metrics(w, problem);
    while trajectory.next_index <= numel(trajectory.samples) && ...
            samples_used >= trajectory.samples(trajectory.next_index)
        index = trajectory.next_index;
        trajectory.relative_error(index) = metrics.relative_error;
        trajectory.relative_residual(index) = metrics.relative_residual;
        trajectory.objective(index) = metrics.objective;
        trajectory.next_index = trajectory.next_index + 1;
    end
end

function trajectory = finalize_trajectory(trajectory, w, problem)
    metrics = evaluate_metrics(w, problem);
    while trajectory.next_index <= numel(trajectory.samples)
        index = trajectory.next_index;
        trajectory.relative_error(index) = metrics.relative_error;
        trajectory.relative_residual(index) = metrics.relative_residual;
        trajectory.objective(index) = metrics.objective;
        trajectory.next_index = trajectory.next_index + 1;
    end

    trajectory = rmfield(trajectory, 'next_index');
end

function metrics = evaluate_metrics(w, problem)
    if any(~isfinite(w))
        metrics.relative_error = problem.divergence_error;
        metrics.relative_residual = problem.divergence_error;
        metrics.objective = problem.divergence_error;
        return;
    end

    residual = problem.bellman_map * w - problem.r_pi;
    relative_error = norm(w - problem.w_star) / problem.w_star_norm;
    relative_residual = norm(residual) / problem.r_pi_norm;
    objective = mean(pseudo_huber_loss( ...
        residual, problem.huber_delta));

    if ~isfinite(relative_error) || ~isfinite(relative_residual) || ...
            ~isfinite(objective)
        metrics.relative_error = problem.divergence_error;
        metrics.relative_residual = problem.divergence_error;
        metrics.objective = problem.divergence_error;
    else
        metrics.relative_error = min(relative_error, problem.divergence_error);
        metrics.relative_residual = min(relative_residual, problem.divergence_error);
        metrics.objective = min(objective, problem.divergence_error);
    end
end

%% ========================================================================
% Plotting and export
% =========================================================================
function fig = plot_case_results(summary, cfg, noise_type, alpha)
    fig = figure('Position', cfg.figure_position, 'Color', 'w');
    axes_handle = axes(fig); 
    hold(axes_handle, 'on');

    line_handles = gobjects(1, numel(cfg.algorithm_order));
    marker_indices = unique(round(linspace(1, ...
        numel(summary.(cfg.algorithm_order{1}).samples), 10)));

    all_lower = [];
    all_upper = [];

    for k = 1:numel(cfg.algorithm_order)
        name = cfg.algorithm_order{k};
        style = cfg.plot_styles.(name);
        method_summary = summary.(name);

        x = method_summary.samples;
        y = max(method_summary.mean_relative_error, cfg.min_positive_y);
        lower = max(method_summary.ci_lower, cfg.min_positive_y);
        upper = max(method_summary.ci_upper, lower);

        fill(axes_handle, [x, fliplr(x)], ...
            [lower, fliplr(upper)], style.color, ...
            'FaceAlpha', 0.08, 'EdgeColor', 'none', ...
            'HandleVisibility', 'off');

        line_handles(k) = semilogy(axes_handle, x, y, ...
            'Color', style.color, ...
            'LineStyle', style.line, ...
            'LineWidth', cfg.line_width, ...
            'Marker', style.marker, ...
            'MarkerIndices', marker_indices, ...
            'MarkerSize', cfg.marker_size, ...
            'MarkerFaceColor', 'w');

        all_lower = [all_lower, lower]; %#ok<AGROW>
        all_upper = [all_upper, upper]; %#ok<AGROW>
    end

    grid(axes_handle, 'on');
    box(axes_handle, 'on');
    axes_handle.GridAlpha = 0.20;
    axes_handle.MinorGridAlpha = 0.10;
    axes_handle.YMinorGrid = 'on';
    axes_handle.FontName = 'Times New Roman';
    axes_handle.FontSize = cfg.font_size;
    axes_handle.LineWidth = 1.0;

    xlabel(axes_handle, 'Number of samples', ...
        'Interpreter', 'latex', 'FontSize', cfg.font_size + 2);
    ylabel(axes_handle, '$\|w_t-w^\star\|/\|w^\star\|$', ...
        'Interpreter', 'latex', 'FontSize', cfg.font_size + 2);

    if strcmpi(noise_type, 'sas')
        title_text = sprintf('$S\\alpha S$ rewards, $\\bar{\\alpha}=%.3g$', alpha);
    else
        title_text = sprintf('Pareto rewards, $\\bar{\\alpha}=%.3g$', alpha);
    end
    title(axes_handle, title_text, 'Interpreter', 'latex', ...
        'FontSize', cfg.font_size + 4, 'FontWeight', 'normal');

    labels = cell(1, numel(cfg.algorithm_order));
    for k = 1:numel(cfg.algorithm_order)
        labels{k} = cfg.algorithm_labels.(cfg.algorithm_order{k});
    end
    legend(axes_handle, line_handles, labels, ...
        'Location', cfg.legend_location, ...
        'FontSize', cfg.legend_font_size, ...
        'Interpreter', 'none', 'NumColumns', 2);

    xlim(axes_handle, [0, cfg.sample_budget]);
    finite_lower = all_lower(isfinite(all_lower) & all_lower > 0);
    finite_upper = all_upper(isfinite(all_upper) & all_upper > 0);
    if ~isempty(finite_lower) && ~isempty(finite_upper)
        y_min = max(cfg.min_positive_y, 0.5 * min(finite_lower));
        y_max = max(1.2, 1.2 * max(finite_upper));
        if y_min < y_max
            ylim(axes_handle, [y_min, y_max]);
        end
    end

    set(fig, 'Renderer', 'painters');
end

function export_academic_figure(fig, base_path, cfg)

    old_units = get(fig, 'Units');
    set(fig, 'Units', 'inches');
    figure_position = get(fig, 'Position');
    set(fig, 'PaperUnits', 'inches', ...
        'PaperPositionMode', 'manual', ...
        'PaperPosition', [0, 0, figure_position(3), figure_position(4)], ...
        'PaperSize', [figure_position(3), figure_position(4)]);
    set(fig, 'Units', old_units);

    if cfg.save_png
        png_path = [base_path, '.png'];
        if exist('exportgraphics', 'file') == 2
            exportgraphics(fig, png_path, ...
                'Resolution', cfg.png_resolution);
        else
            print(fig, png_path, '-dpng', ...
                sprintf('-r%d', cfg.png_resolution));
        end
    end

    if cfg.save_pdf
        pdf_path = [base_path, '.pdf'];

        print(fig, pdf_path, '-dpdf', '-painters', '-bestfit');
    end
end
