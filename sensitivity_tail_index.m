function sensitivity_tail_index()

% Expected source files:
%   convergence_sas_alpha_1p2.mat
%   convergence_sas_alpha_1p5.mat
%   convergence_sas_alpha_1p8.mat
%   convergence_pareto_alpha_1p2.mat
%   convergence_pareto_alpha_1p5.mat
%   convergence_pareto_alpha_1p8.mat


    clc;
    close all;

    cfg = local_config();
    [source_dir, project_root] = locate_results_folder( ...
        cfg.source_folder);

    output_dir = fullfile(project_root, cfg.output_folder);
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    cases = define_cases();
    methods = {'NSCG_M', 'NSCG_S'};
    method_labels = {'NSCG-M', 'NSCG-S'};

    combined = struct();
    for method_index = 1:numel(methods)
        combined.(methods{method_index}) = repmat(empty_curve(), ...
            1, numel(cases));
    end

    fprintf('============================================================\n');
    fprintf('Tail-index sensitivity assembled from existing MAT files\n');
    fprintf('Source folder:\n  %s\n', source_dir);
    fprintf('Output folder:\n  %s\n', output_dir);
    fprintf('NSCG-M and NSCG-S will NOT be rerun.\n');
    fprintf('============================================================\n');

    inherited_plot_cfg = struct();

    for case_index = 1:numel(cases)
        source_path = find_case_file(source_dir, ...
            cases(case_index).noise_type, cases(case_index).alpha);

        loaded = load_required_variables(source_path);
        validate_case_metadata(loaded, cases(case_index), source_path);

        if isempty(fieldnames(inherited_plot_cfg)) && ...
                isfield(loaded, 'case_cfg')
            inherited_plot_cfg = loaded.case_cfg;
        end

        fprintf('\nLoaded: %s\n', source_path);

        for method_index = 1:numel(methods)
            method_name = methods{method_index};
            curve = extract_method_curve(loaded, method_name, ...
                cfg.min_positive_y, source_path);

            curve.noise_type = cases(case_index).noise_type;
            curve.alpha = cases(case_index).alpha;
            curve.label = cases(case_index).label;
            curve.source_file = source_path;
            curve.sample_budget = curve.samples(end);

            combined.(method_name)(case_index) = curve;

            fprintf('  %-7s: points=%d, endpoint=%.3e, final error=%.4e\n', ...
                method_labels{method_index}, numel(curve.samples), ...
                curve.samples(end), curve.mean_relative_error(end));
        end
    end

    cfg = inherit_plot_settings(cfg, inherited_plot_cfg);

    all_endpoints = zeros(1, numel(cases));
    for case_index = 1:numel(cases)
        all_endpoints(case_index) = ...
            combined.NSCG_M(case_index).samples(end);
    end

    if isempty(cfg.common_sample_horizon)
        x_limit = max(all_endpoints);
        fprintf('\nPlotting every stored curve over its full original horizon.\n');
        fprintf('The six source horizons range from %.3e to %.3e samples.\n', ...
            min(all_endpoints), max(all_endpoints));
    else
        if ~isscalar(cfg.common_sample_horizon) || ...
                ~isfinite(cfg.common_sample_horizon) || ...
                cfg.common_sample_horizon <= 0
            error('common_sample_horizon must be empty or a positive scalar.');
        end
        if cfg.common_sample_horizon > min(all_endpoints)
            error(['common_sample_horizon=%.3e exceeds the shortest stored ', ...
                'curve endpoint %.3e.'], ...
                cfg.common_sample_horizon, min(all_endpoints));
        end
        x_limit = cfg.common_sample_horizon;
        fprintf('\nCropping all six curves to the common horizon %.3e.\n', ...
            x_limit);
    end

    [styles, labels] = make_case_styles(cases);

    for method_index = 1:numel(methods)
        method_name = methods{method_index};
        curves = combined.(method_name);

        if ~isempty(cfg.common_sample_horizon)
            for case_index = 1:numel(curves)
                curves(case_index) = trim_curve_to_horizon( ...
                    curves(case_index), cfg.common_sample_horizon);
            end
        end

        fig = plot_tail_sensitivity(curves, labels, styles, cfg, ...
            method_labels{method_index}, x_limit);

        base_path = fullfile(output_dir, ...
            [method_name, '_tail_index_sensitivity_from_mat']);
        save_figure_files(fig, base_path, cfg);

        if ~cfg.keep_figures_open
            close(fig);
        end
    end

    source_folder = source_dir; %#ok<NASGU>
    save(fullfile(output_dir, ...
        'tail_index_sensitivity_from_existing_results.mat'), ...
        'combined', 'cases', 'cfg', 'source_folder', '-v7');

    fprintf('\nFinished. No algorithm was rerun.\n');
    fprintf('Editable FIG, PNG, PDF, and combined MAT files are in:\n  %s\n', ...
        output_dir);
end

%% ========================================================================
% User-editable settings
% =========================================================================
function cfg = local_config()
    cfg.source_folder = 'results_minimal';
    cfg.output_folder = 'results_sensitivity_tail_index_from_mat';

    cfg.common_sample_horizon = [];

    cfg.show_confidence_interval = true;
    cfg.keep_figures_open = true;

    cfg.figure_position = [100, 100, 820, 600];
    cfg.font_size = 16;
    cfg.legend_font_size = 12;
    cfg.line_width = 2.5;
    cfg.marker_size = 7;
    cfg.number_of_markers = 10;
    cfg.ci_face_alpha = 0.055;
    cfg.min_positive_y = 1.0e-8;
    cfg.legend_location = 'best';
    cfg.legend_num_columns = 2;
    cfg.legend_item_token_size = [30, 16];

    cfg.save_fig = true;
    cfg.save_png = true;
    cfg.save_pdf = true;
    cfg.png_resolution = 600;
end

function cases = define_cases()
    alpha_values = [1.2, 1.5, 1.8];
    number_of_alphas = numel(alpha_values);

    cases = repmat(struct( ...
        'noise_type', '', ...
        'alpha', NaN, ...
        'label', ''), 1, 2 * number_of_alphas);

    for k = 1:number_of_alphas
        alpha = alpha_values(k);

        cases(k).noise_type = 'sas';
        cases(k).alpha = alpha;
        cases(k).label = sprintf( ...
            '$\\mathrm{S}\\alpha\\mathrm{S},\\ \\bar{\\alpha}=%.1f$', ...
            alpha);

        cases(number_of_alphas + k).noise_type = 'pareto';
        cases(number_of_alphas + k).alpha = alpha;
        cases(number_of_alphas + k).label = sprintf( ...
            '$\\mathrm{Pareto},\\ \\bar{\\alpha}=%.1f$', alpha);
    end
end

function curve = empty_curve()
    curve = struct( ...
        'noise_type', '', ...
        'alpha', NaN, ...
        'label', '', ...
        'source_file', '', ...
        'sample_budget', NaN, ...
        'samples', [], ...
        'mean_relative_error', [], ...
        'ci_lower', [], ...
        'ci_upper', [], ...
        'number_of_runs', NaN);
end

%% ========================================================================
% Reading existing results
% =========================================================================
function [source_dir, project_root] = locate_results_folder(folder_name)
    script_dir = fileparts(mfilename('fullpath'));
    parent_dir = fileparts(script_dir);
    current_dir = pwd;

    candidates = { ...
        fullfile(script_dir, folder_name), ...
        fullfile(parent_dir, folder_name), ...
        fullfile(current_dir, folder_name)};

    source_dir = '';
    for k = 1:numel(candidates)
        if exist(candidates{k}, 'dir') == 7
            source_dir = candidates{k};
            break;
        end
    end

    if isempty(source_dir)
        error(['Could not find the folder "%s" beside this file, in its ', ...
            'parent directory, or under the current MATLAB directory.'], ...
            folder_name);
    end

    project_root = fileparts(source_dir);
end

function source_path = find_case_file(source_dir, noise_type, alpha)
    alpha_tag = strrep(sprintf('%.6g', alpha), '.', 'p');
    exact_name = sprintf('convergence_%s_alpha_%s.mat', ...
        lower(noise_type), alpha_tag);
    exact_path = fullfile(source_dir, exact_name);

    if exist(exact_path, 'file') == 2
        source_path = exact_path;
        return;
    end

    candidates = dir(fullfile(source_dir, sprintf( ...
        'convergence_%s_alpha_*.mat', lower(noise_type))));

    for k = 1:numel(candidates)
        candidate_path = fullfile(candidates(k).folder, candidates(k).name);
        variables = whos('-file', candidate_path);
        variable_names = {variables.name};
        if ~ismember('alpha', variable_names)
            continue;
        end
        metadata = load(candidate_path, 'alpha');
        tolerance = 100 * eps(max(1, abs(alpha)));
        if isscalar(metadata.alpha) && ...
                abs(metadata.alpha - alpha) <= tolerance
            source_path = candidate_path;
            return;
        end
    end

    error('Missing source MAT file for %s alpha=%.3g in:\n%s', ...
        upper(noise_type), alpha, source_dir);
end

function loaded = load_required_variables(source_path)
    variables = whos('-file', source_path);
    variable_names = {variables.name};

    required_metadata = {'noise_type', 'alpha'};
    for k = 1:numel(required_metadata)
        if ~ismember(required_metadata{k}, variable_names)
            error('Variable "%s" is missing from:\n%s', ...
                required_metadata{k}, source_path);
        end
    end

    variables_to_load = {'noise_type', 'alpha'};

    optional_names = {'summary', 'raw', 'checkpoints', 'case_cfg'};
    for k = 1:numel(optional_names)
        if ismember(optional_names{k}, variable_names)
            variables_to_load{end + 1} = optional_names{k}; %#ok<AGROW>
        end
    end

    if ~ismember('summary', variable_names) && ...
            ~ismember('raw', variable_names)
        error(['Neither SUMMARY nor RAW is available in:\n%s\n', ...
            'At least one of them is required.'], source_path);
    end

    loaded = load(source_path, variables_to_load{:});
end

function validate_case_metadata(loaded, expected_case, source_path)
    stored_noise = loaded.noise_type;
    if isstring(stored_noise)
        stored_noise = char(stored_noise);
    end

    if ~ischar(stored_noise) || ...
            ~strcmpi(strtrim(stored_noise), expected_case.noise_type)
        error('Noise-type mismatch in:\n%s', source_path);
    end

    tolerance = 100 * eps(max(1, abs(expected_case.alpha)));
    if ~isscalar(loaded.alpha) || ...
            abs(loaded.alpha - expected_case.alpha) > tolerance
        error('Tail-index mismatch in:\n%s', source_path);
    end
end

function curve = extract_method_curve( ...
        loaded, method_name, min_positive_y, source_path)

    curve = empty_curve();

    if isfield(loaded, 'summary') && ...
            isfield(loaded.summary, method_name)
        stored = loaded.summary.(method_name);

        if isfield(stored, 'samples')
            samples = stored.samples;
        elseif isfield(loaded, 'checkpoints')
            samples = loaded.checkpoints;
        else
            error('No sample coordinates are stored for %s in:\n%s', ...
                method_name, source_path);
        end

        required_fields = {'mean_relative_error', 'ci_lower', 'ci_upper'};
        has_complete_summary = true;
        for k = 1:numel(required_fields)
            has_complete_summary = has_complete_summary && ...
                isfield(stored, required_fields{k});
        end

        if has_complete_summary
            mean_error = stored.mean_relative_error;
            ci_lower = stored.ci_lower;
            ci_upper = stored.ci_upper;

            number_of_runs = NaN;
            if isfield(loaded, 'raw') && ...
                    isfield(loaded.raw, method_name) && ...
                    isfield(loaded.raw.(method_name), 'relative_error')
                number_of_runs = size( ...
                    loaded.raw.(method_name).relative_error, 1);
            end
        else
            [mean_error, ci_lower, ci_upper, number_of_runs] = ...
                compute_from_raw(loaded, method_name, min_positive_y, ...
                    source_path);
        end
    else
        if isfield(loaded, 'checkpoints')
            samples = loaded.checkpoints;
        else
            error('CHECKPOINTS is required when SUMMARY is unavailable in:\n%s', ...
                source_path);
        end
        [mean_error, ci_lower, ci_upper, number_of_runs] = ...
            compute_from_raw(loaded, method_name, min_positive_y, ...
                source_path);
    end

    samples = reshape(samples, 1, []);
    mean_error = reshape(mean_error, 1, []);
    ci_lower = reshape(ci_lower, 1, []);
    ci_upper = reshape(ci_upper, 1, []);

    number_of_points = numel(samples);
    if numel(mean_error) ~= number_of_points || ...
            numel(ci_lower) ~= number_of_points || ...
            numel(ci_upper) ~= number_of_points
        error('Stored curve lengths are inconsistent for %s in:\n%s', ...
            method_name, source_path);
    end

    if any(~isfinite(samples)) || any(diff(samples) < 0)
        error('The sample coordinates are invalid for %s in:\n%s', ...
            method_name, source_path);
    end

    mean_error = max(mean_error, min_positive_y);
    ci_lower = max(ci_lower, min_positive_y);
    ci_upper = max(ci_upper, ci_lower);

    curve.samples = samples;
    curve.mean_relative_error = mean_error;
    curve.ci_lower = ci_lower;
    curve.ci_upper = ci_upper;
    curve.number_of_runs = number_of_runs;
end

function [mean_error, ci_lower, ci_upper, number_of_runs] = ...
        compute_from_raw(loaded, method_name, min_positive_y, source_path)

    if ~isfield(loaded, 'raw') || ...
            ~isfield(loaded.raw, method_name) || ...
            ~isfield(loaded.raw.(method_name), 'relative_error')
        error('RAW.%s.relative_error is missing from:\n%s', ...
            method_name, source_path);
    end

    errors = loaded.raw.(method_name).relative_error;
    if isempty(errors) || ~ismatrix(errors)
        error('RAW.%s.relative_error is invalid in:\n%s', ...
            method_name, source_path);
    end

    number_of_runs = size(errors, 1);
    mean_error = mean(errors, 1, 'omitnan');

    if number_of_runs > 1
        standard_error = std(errors, 0, 1, 'omitnan') ./ ...
            sqrt(sum(isfinite(errors), 1));
    else
        standard_error = zeros(size(mean_error));
    end

    z_value = 1.959963984540054;
    ci_lower = max(mean_error - z_value .* standard_error, ...
        min_positive_y);
    ci_upper = mean_error + z_value .* standard_error;
end

function cfg = inherit_plot_settings(cfg, source_cfg)

    fallback_names = {'min_positive_y', 'png_resolution'};

    for k = 1:numel(fallback_names)
        name = fallback_names{k};
        local_missing = ~isfield(cfg, name) || isempty(cfg.(name));
        if local_missing && isfield(source_cfg, name)
            cfg.(name) = source_cfg.(name);
        end
    end
end

%% ========================================================================
% Curve processing and plotting
% =========================================================================
function curve = trim_curve_to_horizon(curve, horizon)
    x = curve.samples;
    y = curve.mean_relative_error;
    lower = curve.ci_lower;
    upper = curve.ci_upper;

    keep = x <= horizon;
    if ~any(keep)
        error('No stored points occur at or before horizon %.3e.', horizon);
    end

    x_new = x(keep);
    y_new = y(keep);
    lower_new = lower(keep);
    upper_new = upper(keep);

    if x_new(end) < horizon && horizon < x(end)
        y_horizon = interp1(x, y, horizon, 'linear');
        lower_horizon = interp1(x, lower, horizon, 'linear');
        upper_horizon = interp1(x, upper, horizon, 'linear');

        x_new(end + 1) = horizon;
        y_new(end + 1) = y_horizon;
        lower_new(end + 1) = lower_horizon;
        upper_new(end + 1) = upper_horizon;
    end

    curve.samples = x_new;
    curve.mean_relative_error = y_new;
    curve.ci_lower = lower_new;
    curve.ci_upper = upper_new;
    curve.sample_budget = x_new(end);
end

function [styles, labels] = make_case_styles(cases)

    if numel(cases) ~= 6
        error('The high-contrast style table expects exactly six cases.');
    end

    styles = repmat(struct( ...
        'color', [0, 0, 0], ...
        'line', '-', ...
        'marker', 'o'), 1, 6);

    % S-alpha-S, alpha = 1.2
    styles(1) = struct('color', [0.35 0.35 0.35], ...
        'line', ':',  'marker', 'o');

    % S-alpha-S, alpha = 1.5
    styles(2) = struct('color', [0.90 0.60 0.00], ...
        'line', '--', 'marker', 's');

    % S-alpha-S, alpha = 1.8
    styles(3) = struct('color', [0.00 0.45 0.70], ...
        'line', '-.', 'marker', '^');

    % Pareto, alpha = 1.2
    styles(4) = struct('color', [0.49 0.18 0.56], ...
        'line', '--', 'marker', 'd');

    % Pareto, alpha = 1.5
    styles(5) = struct('color', [0.80 0.40 0.00], ...
        'line', '-',  'marker', 'v');

    % Pareto, alpha = 1.8
    styles(6) = struct('color', [0.00 0.62 0.45], ...
        'line', '--', 'marker', '>');

    labels = {cases.label};
end

function fig = plot_tail_sensitivity( ...
        curves, labels, styles, cfg, method_label, x_limit)

    number_of_curves = numel(curves);
    fig = figure('Position', cfg.figure_position, 'Color', 'w');
    axes_handle = axes(fig);
    hold(axes_handle, 'on');

    line_handles = gobjects(1, number_of_curves);
    all_lower = [];
    all_upper = [];

    plot_data = repmat(struct( ...
        'x', [], 'y', [], 'lower', [], 'upper', []), ...
        1, number_of_curves);

    for k = 1:number_of_curves
        curve = curves(k);
        x = curve.samples;
        y = max(curve.mean_relative_error, cfg.min_positive_y);
        lower = max(curve.ci_lower, cfg.min_positive_y);
        upper = max(curve.ci_upper, lower);

        finite_mask = isfinite(x) & isfinite(y) & ...
            isfinite(lower) & isfinite(upper);

        x = x(finite_mask);
        y = y(finite_mask);
        lower = lower(finite_mask);
        upper = upper(finite_mask);

        if isempty(x)
            error('Curve %d contains no finite points.', k);
        end

        plot_data(k).x = x;
        plot_data(k).y = y;
        plot_data(k).lower = lower;
        plot_data(k).upper = upper;

        all_lower = [all_lower, lower]; %#ok<AGROW>
        all_upper = [all_upper, upper]; %#ok<AGROW>
    end

    if cfg.show_confidence_interval
        for k = 1:number_of_curves
            x = plot_data(k).x;
            lower = plot_data(k).lower;
            upper = plot_data(k).upper;

            fill(axes_handle, [x, fliplr(x)], ...
                [lower, fliplr(upper)], styles(k).color, ...
                'FaceAlpha', cfg.ci_face_alpha, ...
                'EdgeColor', 'none', ...
                'HandleVisibility', 'off');
        end
    end

    for k = 1:number_of_curves
        x = plot_data(k).x;
        y = plot_data(k).y;

        marker_indices = unique(round(linspace( ...
            1, numel(x), min(cfg.number_of_markers, numel(x)))));

        line_handles(k) = semilogy(axes_handle, x, y, ...
            'Color', styles(k).color, ...
            'LineStyle', styles(k).line, ...
            'LineWidth', cfg.line_width, ...
            'Marker', styles(k).marker, ...
            'MarkerIndices', marker_indices, ...
            'MarkerSize', cfg.marker_size, ...
            'MarkerFaceColor', 'w', ...
            'MarkerEdgeColor', styles(k).color);
    end

    uistack(line_handles, 'top');

    grid(axes_handle, 'on');
    box(axes_handle, 'on');
    axes_handle.GridAlpha = 0.20;
    axes_handle.MinorGridAlpha = 0.10;
    axes_handle.YMinorGrid = 'on';
    axes_handle.FontName = 'Times New Roman';
    axes_handle.FontSize = cfg.font_size;
    axes_handle.LineWidth = 1.0;

    xlabel(axes_handle, 'Number of samples', ...
        'Interpreter', 'latex', ...
        'FontSize', cfg.font_size + 2);
    ylabel(axes_handle, '$\|w_t-w^\star\|/\|w^\star\|$', ...
        'Interpreter', 'latex', ...
        'FontSize', cfg.font_size + 2);

    title(axes_handle, sprintf('%s: noise-tail sensitivity', method_label), ...
        'Interpreter', 'none', ...
        'FontSize', cfg.font_size + 3, ...
        'FontWeight', 'normal');

    legend_handle = legend(axes_handle, line_handles, labels, ...
        'Location', cfg.legend_location, ...
        'FontSize', cfg.legend_font_size, ...
        'Interpreter', 'latex', ...
        'NumColumns', cfg.legend_num_columns, ...
        'Box', 'on');

    legend_handle.Color = 'white';
    legend_handle.EdgeColor = [0.25 0.25 0.25];
    legend_handle.LineWidth = 0.8;

    if isprop(legend_handle, 'ItemTokenSize')
        legend_handle.ItemTokenSize = cfg.legend_item_token_size;
    end

    xlim(axes_handle, [0, x_limit]);

    finite_lower = all_lower(isfinite(all_lower) & all_lower > 0);
    finite_upper = all_upper(isfinite(all_upper) & all_upper > 0);
    if ~isempty(finite_lower) && ~isempty(finite_upper)
        y_min = max(cfg.min_positive_y, 0.7 * min(finite_lower));
        y_max = max(1.2, 1.15 * max(finite_upper));
        if y_min < y_max
            ylim(axes_handle, [y_min, y_max]);
        end
    end

    set(fig, 'Renderer', 'painters');
    drawnow;
end

function save_figure_files(fig, base_path, cfg)
    if cfg.save_fig
        savefig(fig, [base_path, '.fig']);
    end

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
            exportgraphics(fig, pdf_path, 'ContentType', 'vector');
        else
            old_units = get(fig, 'Units');
            set(fig, 'Units', 'inches');
            figure_position = get(fig, 'Position');
            set(fig, 'PaperUnits', 'inches', ...
                'PaperPositionMode', 'manual', ...
                'PaperPosition', [0, 0, ...
                    figure_position(3), figure_position(4)], ...
                'PaperSize', [figure_position(3), figure_position(4)]);
            set(fig, 'Units', old_units);
            print(fig, pdf_path, '-dpdf', '-painters', '-bestfit');
        end
    end
end
