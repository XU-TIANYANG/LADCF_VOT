function [features, gparams, feature_info] = init_features_res(features, gparams, is_color_image, img_sample_sz, size_mode)

if nargin < 3
    size_mode = 'same';
end

% Set missing global parameters to default values
if ~isfield(gparams, 'normalize_power')
    gparams.normalize_power = [];
end
if ~isfield(gparams, 'normalize_size')
    gparams.normalize_size = true;
end
if ~isfield(gparams, 'normalize_dim')
    gparams.normalize_dim = false;
end
if ~isfield(gparams, 'square_root_normalization')
    gparams.square_root_normalization = false;
end
if ~isfield(gparams, 'use_gpu')
    gparams.use_gpu = false;
end

% find which features to keep
feat_ind = false(length(features),1);
for n = 1:length(features)
    
    if ~isfield(features{n}.fparams,'useForColor')
        features{n}.fparams.useForColor = true;
    end
    
    if ~isfield(features{n}.fparams,'useForGray')
        features{n}.fparams.useForGray = true;
    end
    
    if (features{n}.fparams.useForColor && is_color_image) || (features{n}.fparams.useForGray && ~is_color_image)
        % keep feature
        feat_ind(n) = true;
    end
end

% remove features that are not used
features = features(feat_ind);

num_features = length(features);

feature_info.min_cell_size = zeros(num_features,1);


% Initialize features by
% - setting the dimension (nDim)
% - specifying if a cell array is returned (is_cell)
% - setting default values of missing feature-specific parameters
% - loading and initializing necessary data (e.g. the lookup table or the network)
for k = 1:length(features)
    if isequal(features{k}.getFeature, @get_fhog)
        if ~isfield(features{k}.fparams, 'nOrients')
            features{k}.fparams.nOrients = 9;
        end
        features{k}.fparams.nDim = 3*features{k}.fparams.nOrients+5-1;
        features{k}.is_cell = false;
        features{k}.is_cnn = false;
        
    elseif isequal(features{k}.getFeature, @get_table_feature)
        table = load(['lookup_tables/' features{k}.fparams.tablename]);
        features{k}.fparams.nDim = size(table.(features{k}.fparams.tablename),2);
        features{k}.is_cell = false;
        features{k}.is_cnn = false;
        
    elseif isequal(features{k}.getFeature, @get_colorspace)
        features{k}.fparams.nDim = 1;
        features{k}.is_cell = false;
        features{k}.is_cnn = false;
        
    elseif isequal(features{k}.getFeature, @get_cnn_layers) || isequal(features{k}.getFeature, @get_OFcnn_layers)
        
        % Set default parameters
        if ~isfield(features{k}.fparams, 'input_size_mode')
            features{k}.fparams.input_size_mode = 'adaptive';
        end
        if ~isfield(features{k}.fparams, 'input_size_scale')
            features{k}.fparams.input_size_scale = 1;
        end
        if ~isfield(features{k}.fparams, 'downsample_factor')
            features{k}.fparams.downsample_factor = ones(1, length(features{k}.fparams.output_layer));
        end
        
        % load the network
        net = dagnn.DagNN.loadobj(load('imagenet-resnet-50-dag.mat'));
        %net.conserveMemory = 0;
        net.mode = 'test';
        features{k}.fparams.output_var = net.getVarIndex(features{k}.fparams.output_var);
        
        layer_names = {};
        total_layer_num = numel(net.layers);
        for ii = 1 : total_layer_num - features{k}.fparams.output_var(end)+1
            layer_names{ii} = net.layers(ii + features{k}.fparams.output_var(end)-1).name;
        end
        net.removeLayer(layer_names);

        % find the dimensionality of each layer
        dim_layer = net.getVarSizes({'data',img_sample_sz});
        features{k}.fparams.nDim = zeros(numel(features{k}.fparams.output_var),1);
        for i = 1:numel(features{k}.fparams.output_var)
            features{k}.fparams.nDim(i) = dim_layer{features{k}.fparams.output_var(i)}(3);
        end
        
        % find the stride of the layers
        net_RF = net.getVarReceptiveFields(1);

        % compute the cell size of the layers (takes down-sampling factor
        % into account)
        features{k}.fparams.cell_size = zeros(numel(features{k}.fparams.output_var),1);
        for i = 1:numel(features{k}.fparams.output_var)
             features{k}.fparams.cell_size(i) = net_RF(features{k}.fparams.output_var(i)).stride(1)...
                 * features{k}.fparams.downsample_factor(i);
        end
        
        % this feature will always return a cell array
        features{k}.is_cell = true;
        features{k}.is_cnn = true;
%         features{k}.augment.blur = 1;
%         features{k}.augment.rotation = 1;
%         features{k}.augment.flip = 1;
%         features{k}.augment.dropout = 1;
        
    elseif isequal(features{k}.getFeature,@get_eitel_cnn)
        features{k}.fparams = make_eitel_feature(features{k}.fparams);
    else
        error('Unknown feature type');
    end
    
    % Set default cell size
    if ~isfield(features{k}.fparams, 'cell_size')
        features{k}.fparams.cell_size = 1;
    end
    
    % Set default penalty
    if ~isfield(features{k}.fparams, 'penalty')
        features{k}.fparams.penalty = zeros(length(features{k}.fparams.nDim),1);
    end
    
    % Find the minimum cell size of each layer
    feature_info.min_cell_size(k) = min(features{k}.fparams.cell_size);
end

% Order the features in increasing minimal cell size
[~, feat_ind] = sort(feature_info.min_cell_size);
features = features(feat_ind);
feature_info.min_cell_size = feature_info.min_cell_size(feat_ind);

% Set feature info
feature_info.dim_block = cell(num_features,1);
feature_info.penalty_block = cell(num_features,1);

for k = 1:length(features)
    % update feature info
    feature_info.dim_block{k} = features{k}.fparams.nDim;
    feature_info.penalty_block{k} = features{k}.fparams.penalty(:);
end
% Feature info for each cell block
feature_info.dim = cell2mat(feature_info.dim_block);
feature_info.penalty = cell2mat(feature_info.penalty_block);

% Find if there is any CNN feature
cnn_feature_ind = -1;
for k = 1:length(features)
    if features{k}.is_cnn
        cnn_feature_ind = k;
    end
end

% This ugly code sets the image sample size to be used for extracting the
% features. It then computes the data size (size of the features) and the
% image support size (the corresponding size in the image).
if cnn_feature_ind > 0
    scale = features{cnn_feature_ind}.fparams.input_size_scale;
    
    new_img_sample_sz = img_sample_sz;
    
    if ~strcmpi(size_mode, 'same') && strcmpi(features{cnn_feature_ind}.fparams.input_size_mode, 'adaptive')
        orig_sz = dim_layer{end}(1:2) / features{cnn_feature_ind}.fparams.downsample_factor(end);
        
        if strcmpi(size_mode, 'exact')
            desired_sz = orig_sz + 1;
        elseif strcmpi(size_mode, 'odd_cells')
            desired_sz = orig_sz + 1 + mod(orig_sz,2);
        end
        
        while desired_sz(1) > dim_layer{end}(1)
            new_img_sample_sz = new_img_sample_sz + [1, 0];
            dim_layer = net.getVarSizes({'data',[round(scale * new_img_sample_sz), 3 1]});
        end
        while desired_sz(2) > dim_layer{end}(2)
            new_img_sample_sz = new_img_sample_sz + [0, 1];
            dim_layer = net.getVarSizes({'data',[round(scale * new_img_sample_sz), 3 1]});
        end
    end
    
    feature_info.img_sample_sz = round(new_img_sample_sz);
    
    if strcmpi(features{cnn_feature_ind}.fparams.input_size_mode, 'adaptive')
        features{cnn_feature_ind}.img_input_sz = feature_info.img_sample_sz;
    else
        features{cnn_feature_ind}.img_input_sz = net.meta.normalization.imageSize(1:2);
    end
    
    % Sample size to be input to the net
    scaled_sample_sz = round(scale * features{cnn_feature_ind}.img_input_sz);
    
    net_stride = net_RF(features{cnn_feature_ind}.fparams.output_var(end)).stride;
    total_feat_sz = dim_layer{end}(1:2);
    
    shrink_number = max(2 * ceil((net_stride(end,:) .* total_feat_sz(end,:) - scaled_sample_sz) ./ (2 * net_stride(end,:))), 0);
    
    deepest_layer_sz = total_feat_sz(end,:) - shrink_number;
    scaled_support_sz = net_stride(end,:) .* deepest_layer_sz;
    
    % Calculate output size for each layer
    cnn_output_sz = round(bsxfun(@rdivide, scaled_support_sz, net_stride));
    features{cnn_feature_ind}.fparams.start_ind = floor((total_feat_sz - cnn_output_sz)/2) + 1;
    features{cnn_feature_ind}.fparams.end_ind = features{cnn_feature_ind}.fparams.start_ind + cnn_output_sz - 1;
    
    feature_info.img_support_sz = round(scaled_support_sz .* feature_info.img_sample_sz ./ scaled_sample_sz);
    
    % Set the input size
    features{cnn_feature_ind}.fparams.net = set_cnnres_input_size(net, feature_info.img_sample_sz);
    
    if gparams.use_gpu
        if isempty(gparams.gpu_id)
            gpuDevice();
        elseif gparams.gpu_id > 0
            gpuDevice(gparams.gpu_id);
        end
        features{cnn_feature_ind}.fparams.net = vl_simplenn_move(features{cnn_feature_ind}.fparams.net, 'gpu');
    end
    if numel(features)>1
        temp = feature_info;
        temp.min_cell_size(cnn_feature_ind) = [];
        max_cell_size = max(feature_info.min_cell_size);
    
    if strcmpi(size_mode, 'same')
        img_sample_sz_s = round(img_sample_sz);
    elseif strcmpi(size_mode, 'exact')
        img_sample_sz_s = round(img_sample_sz / max_cell_size) * max_cell_size;
    elseif strcmpi(size_mode, 'odd_cells')
        new_img_sample_sz = (1 + 2*round(img_sample_sz / (2*max_cell_size))) * max_cell_size;
        
        % Check the size with the largest number of odd dimensions (choices in the
        % third dimension)
        feature_sz_choices = floor(bsxfun(@rdivide, bsxfun(@plus, new_img_sample_sz, reshape(0:max_cell_size-1, 1, 1, [])), feature_info.min_cell_size));
        num_odd_dimensions = sum(sum(mod(feature_sz_choices, 2) == 1, 1), 2);
        [~, best_choice] = max(num_odd_dimensions(:));
        pixels_added = best_choice - 1;
        img_sample_sz_s = round(new_img_sample_sz + pixels_added);
    else
        error('Unknown size_mode');
    end
    
    % Setting the feature size and support size
    %     feature_info.data_sz = floor(bsxfun(@rdivide, feature_info.img_sample_sz, feature_info.min_cell_size));
    feature_info.img_support_sz = img_sample_sz_s;
    end    
else 
    max_cell_size = max(feature_info.min_cell_size);
    
    if strcmpi(size_mode, 'same')
        img_sample_sz_s = round(img_sample_sz);
    elseif strcmpi(size_mode, 'exact')
        img_sample_sz_s = round(img_sample_sz / max_cell_size) * max_cell_size;
    elseif strcmpi(size_mode, 'odd_cells')
        new_img_sample_sz = (1 + 2*round(img_sample_sz / (2*max_cell_size))) * max_cell_size;
        
        % Check the size with the largest number of odd dimensions (choices in the
        % third dimension)
        feature_sz_choices = floor(bsxfun(@rdivide, bsxfun(@plus, new_img_sample_sz, reshape(0:max_cell_size-1, 1, 1, [])), feature_info.min_cell_size));
        num_odd_dimensions = sum(sum(mod(feature_sz_choices, 2) == 1, 1), 2);
        [~, best_choice] = max(num_odd_dimensions(:));
        pixels_added = best_choice - 1;
        img_sample_sz_s = round(new_img_sample_sz + pixels_added);
    else
        error('Unknown size_mode');
    end
    
    % Setting the feature size and support size
    %     feature_info.data_sz = floor(bsxfun(@rdivide, feature_info.img_sample_sz, feature_info.min_cell_size));
    feature_info.img_support_sz = img_sample_sz_s;
end

% Set the sample size and data size for each feature
feature_info.data_sz_block = cell(num_features,1);
feature_info.learning_rate_block = cell(num_features,1);
feature_info.feature_selection_rate_block = cell(num_features,1);
feature_info.feature_is_deep_block = cell(num_features,1);
for k = 1:length(features)
    if features{k}.is_cnn
        % CNN features have a different sample size, since the receptive
        % field is often larger than the support size
        features{k}.img_sample_sz = feature_info.img_sample_sz(:)';
        
        % Set the data size based on the computed output size
        feature_info.data_sz_block{k} = floor(bsxfun(@rdivide, cnn_output_sz, features{k}.fparams.downsample_factor'));
        feature_info.learning_rate_block{k} = repmat(features{k}.fparams.learning_rate,numel(features{k}.fparams.output_var),1);
        feature_info.feature_selection_rate_block{k} = repmat(features{k}.fparams.feature_selection_rate,numel(features{k}.fparams.output_var),1);
        feature_info.feature_is_deep_block{k} = repmat(features{k}.fparams.feature_is_deep,numel(features{k}.fparams.output_var),1);
    else
        % implemented classic features always have the same sample and
        % support size
        features{k}.img_sample_sz = feature_info.img_support_sz(:)';
        features{k}.img_input_sz = features{k}.img_sample_sz;
        
        % Set data size based on cell size
        feature_info.data_sz_block{k} = floor(bsxfun(@rdivide, features{k}.img_sample_sz, features{k}.fparams.cell_size));
        feature_info.learning_rate_block{k} = features{k}.fparams.learning_rate;
        feature_info.feature_selection_rate_block{k} = features{k}.fparams.feature_selection_rate;
        feature_info.feature_is_deep_block{k} = features{k}.fparams.feature_is_deep;
    end
end

feature_info.data_sz = cell2mat(feature_info.data_sz_block);
feature_info.learning_rate = cell2mat(feature_info.learning_rate_block);
feature_info.feature_selection_rate = cell2mat(feature_info.feature_selection_rate_block);
feature_info.feature_is_deep = cell2mat(feature_info.feature_is_deep_block);