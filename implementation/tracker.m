function results = tracker(params)

%% Initialization
% Get sequence info
[seq, im] = get_sequence_info(params.seq);
params = rmfield(params, 'seq');

if isempty(im)
    seq.rect_position = [];
    [seq, results] = get_sequence_results(seq);
    return;
end

% Init regularizer
if strcmpi(seq.format, 'vot')
    if numel(seq.region) > 4,
        seq.rect8 = round(seq.region(:));
        rect8 = seq.rect8;
        x1 = round(min(rect8(1:2:end)));
        x2 = round(max(rect8(1:2:end)));
        y1 = round(min(rect8(2:2:end)));
        y2 = round(max(rect8(2:2:end)));
        seq.init_rect = round([x1, y1, x2 - x1, y2 - y1]);
        seq.target_mask = single(poly2mask(rect8(1:2:end)-seq.init_rect(1), ...
            rect8(2:2:end)-seq.init_rect(2), seq.init_rect(4), seq.init_rect(3)));
        seq.t_b_ratio = sum(seq.target_mask(:))/prod(seq.init_rect([4,3]));
    else
        r = seq.region(:);
        seq.rect8 = [r(1),r(2),r(1)+r(3),r(2),r(1)+r(3),r(2)+r(4),r(1),r(2)+r(4)];
        seq.target_mask = single(ones(seq.region([4,3])));
        seq.t_b_ratio = 1;
    end
end

% Init position
pos = seq.init_pos(:)';
target_sz = seq.init_sz(:)';
params.init_sz = target_sz;

% Feature settings
features = params.t_features;

% Set default parameters
params = init_default_params(params);

% Global feature parameters
if isfield(params, 't_global')
    global_fparams = params.t_global;
else
    global_fparams = [];
end

global_fparams.use_gpu = params.use_gpu;
global_fparams.gpu_id = params.gpu_id;
global_fparams.augment = 0;

% Define data types
if params.use_gpu
    params.data_type = zeros(1, 'single', 'gpuArray');
else
    params.data_type = zeros(1, 'single');
end
params.data_type_complex = complex(params.data_type);

global_fparams.data_type = params.data_type;

init_target_sz = target_sz;

% Check if color image
if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        is_color_image = false;
    else
        is_color_image = true;
    end
else
    is_color_image = false;
end

if size(im,3) > 1 && is_color_image == false
    im = im(:,:,1);
end

params.use_mexResize = true;
global_fparams.use_mexResize = true;

% Calculate search area and initial scale factor
search_area = prod(init_target_sz * params.search_area_scale);
if search_area > params.max_image_sample_size
    currentScaleFactor = sqrt(search_area / params.max_image_sample_size);
elseif search_area < params.min_image_sample_size
    currentScaleFactor = sqrt(search_area / params.min_image_sample_size);
else
    currentScaleFactor = 1.0;
end

% target size at the initial scale
base_target_sz = target_sz / currentScaleFactor;

% window size, taking padding into account
switch params.search_area_shape
    case 'proportional'
        img_sample_sz = floor(base_target_sz * params.search_area_scale);     % proportional area, same aspect ratio as the target
    case 'square'
        img_sample_sz = repmat(sqrt(prod(base_target_sz * params.search_area_scale)), 1, 2); % square area, ignores the target aspect ratio
    case 'fix_padding'
        img_sample_sz = base_target_sz + sqrt(prod(base_target_sz * params.search_area_scale) + (base_target_sz(1) - base_target_sz(2))/4) - sum(base_target_sz)/2; % const padding
    case 'custom'
        img_sample_sz = [base_target_sz(1)*2 base_target_sz(2)*2];
end

[features, global_fparams, feature_info] = init_features_res(features, global_fparams, is_color_image, img_sample_sz, 'odd_cells');

% Set feature info
img_support_sz = feature_info.img_support_sz;
feature_sz = feature_info.data_sz;
num_feature_blocks = size(feature_sz, 1);

% Get feature specific parameters
feature_extract_info = get_feature_extract_info(features);

% Size of the extracted feature maps
feature_sz_cell = permute(mat2cell(feature_sz, ones(1,num_feature_blocks), 2), [2 3 1]);
filter_sz = feature_sz + mod(feature_sz+1, 2);
filter_sz_cell = permute(mat2cell(filter_sz, ones(1,num_feature_blocks), 2), [2 3 1]);

% The size of the label function DFT. Equal to the maximum filter size
[output_sz, k1] = max(filter_sz, [], 1);
params.output_sz = output_sz;
k1 = k1(1);

% Get the remaining block indices
block_inds = 1:num_feature_blocks;
block_inds(k1) = [];

% Pre-computes the grid that is used for socre optimization
ky = circshift(-floor((filter_sz_cell{1}(1) - 1)/2) : ceil((filter_sz_cell{1}(1) - 1)/2), [1, -floor((filter_sz_cell{1}(1) - 1)/2)]);
kx = circshift(-floor((filter_sz_cell{1}(2) - 1)/2) : ceil((filter_sz_cell{1}(2) - 1)/2), [1, -floor((filter_sz_cell{1}(2) - 1)/2)])';
newton_iterations = params.newton_iterations;

% Construct the Gaussian label function
yf = cell(num_feature_blocks, 1);
for i = 1:num_feature_blocks
    sz = filter_sz_cell{i};
    output_sigma_factor = params.output_sigma_factor(feature_info.feature_is_deep(i)+1);
    output_sigma = sqrt(prod(floor(base_target_sz)))*feature_sz_cell{i}./img_support_sz* output_sigma_factor;
    rg           = circshift(-floor((sz(1)-1)/2):ceil((sz(1)-1)/2), [0 -floor((sz(1)-1)/2)]);
    cg           = circshift(-floor((sz(2)-1)/2):ceil((sz(2)-1)/2), [0 -floor((sz(2)-1)/2)]);
    [rs, cs]     = ndgrid(rg,cg);
    y            = exp(-0.5 * (((rs.^2 + cs.^2) / mean(output_sigma)^2)));
    yf{i}           = fft2(y);
end

% Compute the cosine windows
cos_window = cellfun(@(sz) hann(sz(1)+2)*hann(sz(2)+2)', feature_sz_cell, 'uniformoutput', false);
cos_window = cellfun(@(cos_window) cast(cos_window(2:end-1,2:end-1), 'like', params.data_type), cos_window, 'uniformoutput', false);

% Define initial regularizer
mask_window = cell(num_feature_blocks, 1);
mask_search_window = ones(round(currentScaleFactor*img_support_sz)) * params.mask_window_min;
target_mask = seq.target_mask;
target_mask_range = zeros(2, 2);
for j = 1:2
    target_mask_range(j,:) = [0, size(target_mask,j) - 1] - floor(size(target_mask,j) / 2);
end
mask_center = floor((size(mask_search_window) + 1)/ 2) + mod(size(mask_search_window) + 1,2);
target_h = (mask_center(1)+ target_mask_range(1,1)) : (mask_center(1) + target_mask_range(1,2));
target_w = (mask_center(2)+ target_mask_range(2,1)) : (mask_center(2) + target_mask_range(2,2));
mask_search_window(target_h, target_w) = target_mask;
for i = 1:num_feature_blocks
    mask_window{i} = mexResize(mask_search_window, filter_sz_cell{i}, 'auto');
end
params.mask_window = mask_window;

% Use the translation filter to estimate the scale
nScales = params.number_of_scales;
scale_step = params.scale_step;
scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2));
scaleFactors = scale_step .^ scale_exp;

if nScales > 0
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ img_support_sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
end

seq.time = 0;
det_sample_pos = pos;
scores_fs_feat = cell(1,1,num_feature_blocks);
while true
    % Read image
    if seq.frame > 0
        [seq, im] = get_sequence_frame(seq);
        if isempty(im)
            break;
        end
        if size(im,3) > 1 && is_color_image == false
            im = im(:,:,1);
        end
    else
        seq.frame = 1;
    end
    
    tic();
    
    %% Target localization step
    if seq.frame > 1
        old_pos = inf(size(pos));
        iter = 1;
        
        %translation search
        while iter <= params.refinement_iterations && any(old_pos ~= pos)
            % Extract features at multiple resolutions
            sample_pos = round(pos);
            det_sample_pos = sample_pos;
            sample_scale = currentScaleFactor*scaleFactors;
            xt = extract_features(im, sample_pos, sample_scale, features, global_fparams, feature_extract_info);
            
            % Do windowing of features
            xtw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xt, cos_window, 'uniformoutput', false);
            
            % Compute the fourier series
            xtf = cellfun(@fft2, xtw, 'uniformoutput', false);
            %xtf = project_sample(xtf, projection_matrix);

            scores_fs_sum_handcrafted = 0;
            scores_fs_sum_deep = 0;
            dim_handcrafted = 0;
            dim_deep = 0;
            for k = [k1 block_inds]
                if feature_info.feature_is_deep(k) == 0
                    scores_fs_feat{k} = gather(sum(bsxfun(@times, conj(theta_model_f{k}), xtf{k}), 3));
                    scores_fs_feat{k} = resizeDFT2(scores_fs_feat{k}, output_sz);
                    scores_fs_sum_handcrafted = scores_fs_sum_handcrafted +  scores_fs_feat{k};
                    dim_handcrafted = dim_handcrafted + feature_info.dim(k);
                else
                    scores_fs_feat{k} = gather(sum(bsxfun(@times, conj(theta_model_f{k}), xtf{k}), 3));
                    scores_fs_feat{k} = resizeDFT2(scores_fs_feat{k}, output_sz);
                    scores_fs_sum_deep = scores_fs_sum_deep +  scores_fs_feat{k};
                    dim_deep = dim_deep + feature_info.dim(k);
                end
            end
            
            scores_fs_handcrafted = permute(gather(scores_fs_sum_handcrafted), [1 2 4 3]);
            scores_fs_deep = permute(gather(scores_fs_sum_deep), [1 2 4 3]);
            response_handcrafted = ifft2(scores_fs_handcrafted, 'symmetric');
            response_deep = ifft2(scores_fs_deep, 'symmetric');
            
            [disp_row, disp_col, sind] = resp_newton(response_handcrafted, response_deep,...
                scores_fs_handcrafted, scores_fs_deep , newton_iterations, ky, kx, output_sz);
            
            % Compute the translation vector in pixel-coordinates and round
            % to the closest integer pixel.
            translation_vec = [disp_row, disp_col] .* (img_support_sz./output_sz) * currentScaleFactor * scaleFactors(sind);
            scale_change_factor = scaleFactors(sind);
            
            % update position
            old_pos = pos;
            if sum(isnan(translation_vec))
                pos = sample_pos;
            else
                pos = sample_pos + translation_vec;
            end
            
            if params.clamp_position
                pos = max([1 1], min([size(im,1) size(im,2)], pos));
            end
            
            % Update the scale
            currentScaleFactor = currentScaleFactor * scale_change_factor;
            
            % Adjust to make sure we are not to large or to small
            if currentScaleFactor < min_scale_factor
                currentScaleFactor = min_scale_factor;
            elseif currentScaleFactor > max_scale_factor
                currentScaleFactor = max_scale_factor;
            end
            
            iter = iter + 1;
        end
    end
    
    %% Model update step
    if seq.frame == 1
        sample_pos = round(pos);
        xl = extract_features(im, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info);
        
        % do windowing of features
        xlw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl, cos_window, 'uniformoutput', false);
        
        % compute the fourier series
        xlf = cellfun(@fft2, xlw, 'uniformoutput', false);

        [theta_model_f, theta_pre_f, ~] = train_theta(xlf, feature_info, yf, seq, params);
    else
        % extract image region for training sample
        sample_pos = round(pos);
        xl = extract_features(im, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info);
        
        % do windowing of features
        xlw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl, cos_window, 'uniformoutput', false);
        
        % compute the fourier series
        xlf = cellfun(@fft2, xlw, 'uniformoutput', false);
        
        [theta_model_f, theta_pre_f, ~] = train_theta(xlf, feature_info, yf, seq, params, theta_pre_f, theta_model_f);
    end
    % Update the target size (only used for computing output box)
    target_sz = base_target_sz * currentScaleFactor;
    
    %save position and calculate FPS
    tracking_result.center_pos = double(pos);
    tracking_result.target_size = double(target_sz);
    seq = report_tracking_result(seq, tracking_result);  
    seq.time = seq.time + toc();
end

[seq, results] = get_sequence_results(seq);

