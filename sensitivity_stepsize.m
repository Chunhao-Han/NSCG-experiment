function sensitivity_stepsize()
% Step-size sensitivity of NSCG-M and NSCG-S.
% Five step sizes are tested: {1e-3, 3e-3, 5e-3, 1e-2}.

    clc;
    close all;

    cfg = base_config();
    cfg.output_folder = 'results_sensitivity_stepsize';

    noise_type = 'sas';
    alpha = 1.5;

    %% Method-specific sample budgets
    cfg.sample_budget_NSCG_M = 5.0e8;
    cfg.sample_budget_NSCG_S = 3.0e8;

    eta_values = [1.0e-3, 3.0e-3, 5.0e-3, 1.0e-2];

    validate_common_config(cfg);

    project_root = fileparts(mfilename('fullpath'));
    output_dir = fullfile(project_root, cfg.output_folder);

    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    methods = {'NSCG_M', 'NSCG_S'};
    method_labels = {'NSCG-M', 'NSCG-S'};
    number_of_candidates = numel(eta_values);

    budget_by_method = struct();
    budget_by_method.NSCG_M = cfg.sample_budget_NSCG_M;
    budget_by_method.NSCG_S = cfg.sample_budget_NSCG_S;

    checkpoints_by_method = struct();
    for method_index = 1:numel(methods)
        method_name = methods{method_index};
        checkpoints_by_method.(method_name) = make_record_checkpoints( ...
            budget_by_method.(method_name), cfg.num_record_points);
    end

    base_mdp = build_base_mdp(cfg);

    problem_bank = cell(cfg.num_runs, 1);
    problem_statistics = zeros(cfg.num_runs, 4);

    for run = 1:cfg.num_runs
        reward_seed = make_reward_seed( ...
            cfg, noise_type, alpha, run);
        problem = build_run_problem( ...
            base_mdp, cfg, noise_type, alpha, reward_seed);

        problem_bank{run} = problem;
        problem_statistics(run, :) = [ ...
            problem.w_star_norm, ...
            min(problem.reward_table(:)), ...
            median(problem.reward_table(:)), ...
            max(problem.reward_table(:))];
    end

    fprintf('============================================================\n');
    fprintf('Step-size sensitivity: NSCG-M and NSCG-S\n');
    fprintf('Noise: symmetric S-alpha-S, alpha=1.5\n');
    fprintf('NSCG-M sample budget: %.0f\n', ...
        cfg.sample_budget_NSCG_M);
    fprintf('NSCG-S sample budget: %.0f\n', ...
        cfg.sample_budget_NSCG_S);
    fprintf('Runs per curve: %d\n', cfg.num_runs);
    fprintf('Testing %d step sizes.\n', number_of_candidates);
    fprintf('============================================================\n');

    results = struct();

    for method_index = 1:numel(methods)
        method_name = methods{method_index};
        method_budget = budget_by_method.(method_name);
        checkpoints = checkpoints_by_method.(method_name);

        candidate_results = repmat(struct( ...
            'eta', NaN, ...
            'sample_budget', method_budget, ...
            'raw', [], ...
            'summary', []), ...
            1, number_of_candidates);

        fprintf('\n============================================================\n');
        fprintf('%s\n', method_labels{method_index});
        fprintf('Sample budget: %.0f\n', method_budget);
        fprintf('============================================================\n');

        for candidate_index = 1:number_of_candidates
            eta = eta_values(candidate_index);
            raw = initialize_method_raw( ...
                cfg.num_runs, numel(checkpoints));

            if strcmp(method_name, 'NSCG_M')
                alg = cfg.alg.NSCG_M;
            else
                alg = cfg.alg.NSCG_S;
            end

            alg.eta = eta;
            fprintf('\neta=%.4g\n', eta);

            for run = 1:cfg.num_runs
                problem = problem_bank{run};
                algorithm_seed = make_algorithm_run_seed( ...
                    cfg, noise_type, alpha, run);

                start_time = tic;

                if strcmp(method_name, 'NSCG_M')
                    trajectory = run_nscg_m( ...
                        problem, alg, method_budget, ...
                        checkpoints, algorithm_seed);
                else
                    trajectory = run_nscg_s( ...
                        problem, alg, method_budget, ...
                        checkpoints, algorithm_seed);
                end

                elapsed_time = toc(start_time);

                raw.relative_error(run, :) = ...
                    trajectory.relative_error;
                raw.relative_residual(run, :) = ...
                    trajectory.relative_residual;
                raw.objective(run, :) = trajectory.objective;
                raw.runtime(run) = elapsed_time;

                fprintf(['  Run %2d/%2d: final error=%9.3e, ', ...
                    'time=%8.2fs\n'], ...
                    run, cfg.num_runs, ...
                    trajectory.relative_error(end), elapsed_time);
            end

            candidate_results(candidate_index).eta = eta;
            candidate_results(candidate_index).sample_budget = ...
                method_budget;
            candidate_results(candidate_index).raw = raw;
            candidate_results(candidate_index).summary = ...
                aggregate_method(raw, checkpoints, cfg);

            mean_final_error = ...
                candidate_results(candidate_index).summary.mean_relative_error(end);
            fprintf('  Mean final error: %.4e\n', mean_final_error);
        end

        results.(method_name) = candidate_results;

        if cfg.save_mat
            save(fullfile(output_dir, ...
                'stepsize_sensitivity_partial.mat'), ...
                'results', 'cfg', 'noise_type', 'alpha', ...
                'eta_values', 'budget_by_method', ...
                'checkpoints_by_method', ...
                'problem_statistics', '-v7.3');
        end
    end

    styles = make_generic_styles(number_of_candidates);
    labels = cell(1, number_of_candidates);

    for candidate_index = 1:number_of_candidates
        labels{candidate_index} = ...
            format_eta_label(eta_values(candidate_index));
    end

    summaries = cell(1, number_of_candidates);
    for candidate_index = 1:number_of_candidates
        summaries{candidate_index} = ...
            results.NSCG_M(candidate_index).summary;
    end

    fig = plot_sensitivity_curves( ...
        summaries, labels, styles, cfg, ...
        'NSCG-M: step-size sensitivity', ...
        cfg.sample_budget_NSCG_M);

    save_sensitivity_figure(fig, ...
        fullfile(output_dir, 'NSCG_M_stepsize_sensitivity'), cfg);
    close(fig);

    for candidate_index = 1:number_of_candidates
        summaries{candidate_index} = ...
            results.NSCG_S(candidate_index).summary;
    end

    fig = plot_sensitivity_curves( ...
        summaries, labels, styles, cfg, ...
        'NSCG-S: step-size sensitivity', ...
        cfg.sample_budget_NSCG_S);

    save_sensitivity_figure(fig, ...
        fullfile(output_dir, 'NSCG_S_stepsize_sensitivity'), cfg);
    close(fig);

    if cfg.save_mat
        save(fullfile(output_dir, ...
            'stepsize_sensitivity_results.mat'), ...
            'results', 'cfg', 'noise_type', 'alpha', ...
            'eta_values', 'budget_by_method', ...
            'checkpoints_by_method', ...
            'problem_statistics', '-v7.3');
    end

    fprintf('\nAll step-size sensitivity results were written to:\n  %s\n', ...
        output_dir);
end

function label = format_eta_label(eta)

    if ~(isscalar(eta) && isnumeric(eta) && isfinite(eta) && eta > 0)
        error('eta must be a positive finite scalar.');
    end

    log_eta = log10(eta);
    rounded_log_eta = round(log_eta);

    if abs(log_eta - rounded_log_eta) <= 1.0e-12
        exponent = rounded_log_eta;
    else
        exponent = floor(log_eta);
    end

    mantissa = eta / (10 ^ exponent);
    rounded_mantissa = round(mantissa);

    if abs(mantissa - rounded_mantissa) <= 1.0e-12
        mantissa = rounded_mantissa;
    end

    if abs(mantissa - 1) <= 1.0e-12
        label = sprintf('$\\eta=10^{%d}$', exponent);
    else
        label = sprintf('$\\eta=%g\\times10^{%d}$', ...
            mantissa, exponent);
    end
end

%% ========================================================================
% Shared configuration and experiment utilities
% =========================================================================
function cfg = base_config()

    cfg.problem_seed = 20260824;
    cfg.reward_seed = 31817;
    cfg.base_seed = 73129;

    cfg.num_runs = 25;
    cfg.num_record_points = 260;

    cfg.S = 100;
    cfg.A = 3;
    cfg.Sp = 4;
    cfg.gamma = 0.90;

    cfg.sas_c = 5.0;

    cfg.pareto_xm = 3.0;

    cfg.huber_delta = 1.0;
    cfg.divergence_error = 1.0e6;

    cfg.alg.NSCG_M.eta = 1.0e-2;
    cfg.alg.NSCG_M.g_batch = 50;
    cfg.alg.NSCG_M.J_batch = 50;

    cfg.alg.NSCG_S.eta = 1.0e-2;
    cfg.alg.NSCG_S.tau = 4;
    cfg.alg.NSCG_S.outer_g_batch = 50;
    cfg.alg.NSCG_S.outer_J_batch = 50;
    cfg.alg.NSCG_S.inner_g_batch = 10;
    cfg.alg.NSCG_S.inner_J_batch = 10;
    cfg.alg.NSCG_S.project_J = false;
    cfg.alg.NSCG_S.J_radius_multiplier = 100;

    % Output and plotting.
    cfg.save_mat = true;
    cfg.save_fig = true;
    cfg.save_png = true;
    cfg.save_pdf = true;
    cfg.png_resolution = 600;

    cfg.figure_position = [100, 100, 900, 650];
    cfg.font_size = 17;
    cfg.legend_font_size = 13;
    cfg.line_width = 2.5;
    cfg.marker_size = 7;
    cfg.min_positive_y = 1.0e-8;
    cfg.legend_location = 'best';
end

function validate_common_config(cfg)
    assert(cfg.S >= 2 && cfg.A >= 1 && cfg.Sp >= 1, ...
        'Invalid MDP dimensions.');
    assert(cfg.Sp <= cfg.S, 'Sp must not exceed S.');
    assert(cfg.gamma > 0 && cfg.gamma < 1, ...
        'gamma must lie in (0,1).');
    assert(cfg.num_runs >= 1 && cfg.num_runs == floor(cfg.num_runs), ...
        'cfg.num_runs must be a positive integer.');
    assert(cfg.num_record_points >= 20, ...
        'cfg.num_record_points must be at least 20.');
    assert(isscalar(cfg.sample_budget_NSCG_M) && ...
        isnumeric(cfg.sample_budget_NSCG_M) && ...
        isfinite(cfg.sample_budget_NSCG_M) && ...
        cfg.sample_budget_NSCG_M >= 1, ...
        'cfg.sample_budget_NSCG_M must be a positive finite scalar.');

    assert(isscalar(cfg.sample_budget_NSCG_S) && ...
        isnumeric(cfg.sample_budget_NSCG_S) && ...
        isfinite(cfg.sample_budget_NSCG_S) && ...
        cfg.sample_budget_NSCG_S >= 1, ...
        'cfg.sample_budget_NSCG_S must be a positive finite scalar.');
    assert(cfg.huber_delta > 0 && isfinite(cfg.huber_delta), ...
        'cfg.huber_delta must be positive and finite.');

    assert(cfg.alg.NSCG_M.eta > 0, ...
        'NSCG-M step size must be positive.');
    assert(is_positive_integer(cfg.alg.NSCG_M.g_batch) && ...
        is_positive_integer(cfg.alg.NSCG_M.J_batch), ...
        'NSCG-M batch sizes must be positive integers.');

    assert(cfg.alg.NSCG_S.eta > 0, ...
        'NSCG-S step size must be positive.');
    assert(is_positive_integer(cfg.alg.NSCG_S.tau) && ...
        is_positive_integer(cfg.alg.NSCG_S.outer_g_batch) && ...
        is_positive_integer(cfg.alg.NSCG_S.outer_J_batch) && ...
        is_positive_integer(cfg.alg.NSCG_S.inner_g_batch) && ...
        is_positive_integer(cfg.alg.NSCG_S.inner_J_batch), ...
        'NSCG-S tau and batch sizes must be positive integers.');
    assert(~cfg.alg.NSCG_S.project_J, ...
        'This sensitivity code assumes project_J=false.');
end

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

function problem = build_run_problem( ...
        base, cfg, noise_type, alpha, reward_seed)

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
            values = draw_centered_pareto( ...
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
        count = sum(unfinished);
        U = pi .* (rand(count, 1) - 0.5);
        W = -log(max(rand(count, 1), realmin('double')));

        standard_values = sin(alpha .* U) ./ ...
            (cos(U).^(1 / alpha)) .* ...
            (cos((1 - alpha) .* U) ./ W).^((1 - alpha) / alpha);
        candidate = sigma .* standard_values;

        finite_mask = isfinite(candidate);
        indices = find(unfinished);
        values(indices(finite_mask)) = candidate(finite_mask);
        unfinished(indices(finite_mask)) = false;
    end
end

function values = draw_centered_pareto(n, alpha, x_m)

    assert(alpha > 1 && alpha <= 2, ...
        'The Pareto tail index must lie in (1,2].');
    assert(x_m > 0, 'The Pareto threshold x_m must be positive.');

    raw = zeros(n, 1);
    unfinished = true(n, 1);

    while any(unfinished)
        count = sum(unfinished);
        u = max(rand(count, 1), realmin('double'));
        candidate = x_m .* u.^(-1 / alpha);

        finite_mask = isfinite(candidate);
        indices = find(unfinished);
        raw(indices(finite_mask)) = candidate(finite_mask);
        unfinished(indices(finite_mask)) = false;
    end

    values = raw - alpha * x_m / (alpha - 1);
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

function checkpoints = make_record_checkpoints(sample_budget, requested_points)

    requested_points = max(20, round(requested_points));
    early_points = max(10, round(0.35 * requested_points));
    late_points = max(10, requested_points - early_points);
    split_budget = max(10, round(0.08 * sample_budget));

    early = round(logspace(0, log10(split_budget), early_points));
    late = round(linspace(split_budget, sample_budget, late_points));
    checkpoints = unique([0, early, late, sample_budget]);
end

function raw = initialize_method_raw(number_of_runs, number_of_points)
    raw.relative_error = zeros(number_of_runs, number_of_points);
    raw.relative_residual = zeros(number_of_runs, number_of_points);
    raw.objective = zeros(number_of_runs, number_of_points);
    raw.runtime = zeros(number_of_runs, 1);
end

function summary = aggregate_method(raw, checkpoints, cfg)

    z_value = 1.959963984540054;
    number_of_runs = size(raw.relative_error, 1);

    mean_error = mean(raw.relative_error, 1);
    mean_residual = mean(raw.relative_residual, 1);
    mean_objective = mean(raw.objective, 1);

    if number_of_runs > 1
        standard_error = std(raw.relative_error, 0, 1) ./ ...
            sqrt(number_of_runs);
    else
        standard_error = zeros(size(mean_error));
    end

    summary.samples = checkpoints;
    summary.mean_relative_error = mean_error;
    summary.ci_lower = max( ...
        mean_error - z_value .* standard_error, cfg.min_positive_y);
    summary.ci_upper = mean_error + z_value .* standard_error;
    summary.mean_relative_residual = mean_residual;
    summary.mean_objective = mean_objective;
    summary.mean_runtime = mean(raw.runtime);
end

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
        J_weights = estimate_J_weights( ...
            problem, alg.outer_J_batch);
        samples_used = samples_used + outer_cost;

        gradient_estimate = apply_J_weights( ...
            problem, J_weights, ...
            pseudo_huber_outer_gradient(g_estimate, problem));
        w_current = w_outer - alg.eta .* ...
            safe_normalize(gradient_estimate);

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

            consume_J_difference_samples( ...
                problem, alg.inner_J_batch);

            samples_used = samples_used + inner_cost;

            gradient_estimate = apply_J_weights( ...
                problem, J_weights, ...
                pseudo_huber_outer_gradient(g_estimate, problem));
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

function difference = estimate_g_difference( ...
        problem, w_new, w_old, batch_size)

    states = randi(problem.S, batch_size, 1);
    counts = accumarray(states, 1, [problem.S, 1], @sum, 0);
    displacement = w_new - w_old;

    first_difference = problem.value_map * displacement;
    second_difference = problem.continuation_map * displacement;
    scale = problem.S / batch_size;

    difference = zeros(2 * problem.S, 1);
    difference(1:2:end) = ...
        scale .* counts .* first_difference;
    difference(2:2:end) = ...
        scale .* counts .* second_difference;
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
    objective = mean(pseudo_huber_loss(residual, problem.huber_delta));

    if ~isfinite(relative_error) || ...
            ~isfinite(relative_residual) || ~isfinite(objective)
        metrics.relative_error = problem.divergence_error;
        metrics.relative_residual = problem.divergence_error;
        metrics.objective = problem.divergence_error;
    else
        metrics.relative_error = min( ...
            relative_error, problem.divergence_error);
        metrics.relative_residual = min( ...
            relative_residual, problem.divergence_error);
        metrics.objective = min( ...
            objective, problem.divergence_error);
    end
end

function direction = safe_normalize(vector)

    if isempty(vector)
        direction = vector;
        return;
    end

    infinite_mask = isinf(vector);
    if any(infinite_mask)
        direction = zeros(size(vector));
        signs = sign(vector(infinite_mask));
        direction(infinite_mask) = ...
            signs ./ sqrt(sum(infinite_mask));
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

function flag = has_diverged(w)
    flag = any(~isfinite(w)) || max(abs(w)) > 1.0e100;
end

function flag = is_positive_integer(value)
    flag = isscalar(value) && isnumeric(value) && isfinite(value) && ...
        value >= 1 && value == floor(value);
end

function styles = make_generic_styles(number_of_curves)

    colors = lines(number_of_curves);
    line_styles = {'-', '--', '-.', ':', '-', '--', '-.', ':'};
    markers = {'o', 's', '^', 'd', 'v', '>', '<', 'p'};

    styles = repmat(struct('color', [], 'line', '', 'marker', ''), ...
        1, number_of_curves);

    for k = 1:number_of_curves
        styles(k).color = colors(k, :);
        styles(k).line = line_styles{1 + mod(k - 1, numel(line_styles))};
        styles(k).marker = markers{1 + mod(k - 1, numel(markers))};
    end
end

function fig = plot_sensitivity_curves( ...
        summaries, labels, styles, cfg, title_text, x_limit)

    number_of_curves = numel(summaries);
    assert(numel(labels) == number_of_curves && ...
        numel(styles) == number_of_curves, ...
        'summaries, labels, and styles must have equal lengths.');

    fig = figure('Position', cfg.figure_position, 'Color', 'w');
    axes_handle = axes(fig);
    hold(axes_handle, 'on');

    line_handles = gobjects(1, number_of_curves);
    all_lower = [];
    all_upper = [];

    for k = 1:number_of_curves
        method_summary = summaries{k};
        style = styles(k);

        x = method_summary.samples;
        y = max(method_summary.mean_relative_error, ...
            cfg.min_positive_y);
        lower = max(method_summary.ci_lower, cfg.min_positive_y);
        upper = max(method_summary.ci_upper, lower);

        fill(axes_handle, [x, fliplr(x)], ...
            [lower, fliplr(upper)], style.color, ...
            'FaceAlpha', 0.07, ...
            'EdgeColor', 'none', ...
            'HandleVisibility', 'off');

        marker_indices = unique(round(linspace( ...
            1, numel(x), min(10, numel(x)))));

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
    axes_handle.LineWidth = 1.1;

    xlabel(axes_handle, 'Number of samples', ...
        'Interpreter', 'latex', ...
        'FontSize', cfg.font_size + 2);
    ylabel(axes_handle, ...
        '$\|w_t-w^\star\|/\|w^\star\|$', ...
        'Interpreter', 'latex', ...
        'FontSize', cfg.font_size + 2);
    title(axes_handle, title_text, ...
        'Interpreter', 'none', ...
        'FontSize', cfg.font_size + 3, ...
        'FontWeight', 'normal');

    legend(axes_handle, line_handles, labels, ...
        'Location', cfg.legend_location, ...
        'FontSize', cfg.legend_font_size, ...
        'Interpreter', 'latex', ...
        'NumColumns', 2);

    xlim(axes_handle, [0, x_limit]);

    finite_lower = all_lower(isfinite(all_lower) & all_lower > 0);
    finite_upper = all_upper(isfinite(all_upper) & all_upper > 0);

    if ~isempty(finite_lower) && ~isempty(finite_upper)
        y_min = max(cfg.min_positive_y, ...
            0.5 * min(finite_lower));
        y_max = max(1.2, 1.2 * max(finite_upper));

        if y_min < y_max
            ylim(axes_handle, [y_min, y_max]);
        end
    end

    set(fig, 'Renderer', 'painters');
end

function save_sensitivity_figure(fig, base_path, cfg)

    if cfg.save_fig
        savefig(fig, [base_path, '.fig']);
    end

    old_units = get(fig, 'Units');
    set(fig, 'Units', 'inches');
    figure_position = get(fig, 'Position');

    set(fig, ...
        'PaperUnits', 'inches', ...
        'PaperPositionMode', 'manual', ...
        'PaperPosition', ...
            [0, 0, figure_position(3), figure_position(4)], ...
        'PaperSize', ...
            [figure_position(3), figure_position(4)]);

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

        if exist('exportgraphics', 'file') == 2
            exportgraphics(fig, pdf_path, ...
                'ContentType', 'vector');
        else
            print(fig, pdf_path, ...
                '-dpdf', '-painters', '-bestfit');
        end
    end
end
