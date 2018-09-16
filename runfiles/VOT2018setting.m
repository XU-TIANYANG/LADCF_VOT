function results = VOT2018setting(seq, res_path, bSaveImage, parameters)

% set features
hog_params.cell_size = 6;
hog_params.nDim = 31;
hog_params.learning_rate = 0.95;
hog_params.feature_selection_rate = 0.05;
hog_params.feature_is_deep = false;

cn_params.tablename = 'CNnorm';
cn_params.useForGray = false;
cn_params.cell_size = 4;
cn_params.nDim = 10;
cn_params.learning_rate = 0.95;
cn_params.feature_selection_rate = 0.05;
cn_params.feature_is_deep = false;

grayscale_params.colorspace='gray';
grayscale_params.cell_size = 4;
grayscale_params.useForColor = false;
grayscale_params.learning_rate = 0.95;
grayscale_params.feature_selection_rate = 0.05;
grayscale_params.feature_is_deep = false;
params.t_global.cell_size = 4;           

cnn_params.nn_name = 'imagenet-resnet-50-dag.mat'; 
cnn_params.output_var = {'res4ex'};    
cnn_params.downsample_factor = [1];           
cnn_params.input_size_mode = 'adaptive';       
cnn_params.input_size_scale = 1;       
cnn_params.learning_rate = 0.15;
cnn_params.feature_selection_rate = 0.15;
cnn_params.feature_is_deep = true;
cnn_params.augment.blur = 1;
cnn_params.augment.rotation = 1;
cnn_params.augment.flip = 1;

% Configure features 
params.t_features = {
    struct('getFeature',@get_cnn_layers, 'fparams',cnn_params),...
    struct('getFeature',@get_colorspace, 'fparams',grayscale_params),...    
    struct('getFeature',@get_fhog,'fparams',hog_params),...
    struct('getFeature',@get_table_feature, 'fparams',cn_params),...
};

% Image sample parameters
params.search_area_shape = 'square';    % The shape of the samples
params.search_area_scale = 4.5;         % The scaling of the target size 
params.min_image_sample_size = 200^2;   % Minimum area of original image samples
params.max_image_sample_size = 250^2;   % Maximum area of original image samples

% Define initial mask
params.mask_window_min = 1e-3;           % the minimum value of the initial feature selection window

% Detection parameters
params.refinement_iterations = 1;       % Number of iterations used to refine the resulting position
params.newton_iterations = 5;           % The number of Newton iterations used for optimizing the detection score
params.clamp_position = false;          % Clamp the target position to be inside the image

% Learning parameters
params.output_sigma_factor = [1/16 1/3];	% Sigma of the label gaussian function 
params.temporal_consistency_factor = [15 13];   % The temporal consistency parameters
params.max_iterations = 2;
params.init_penalty_factor = 1;
params.penalty_scale_step = 10;

% Scale parameters for the translation model
params.number_of_scales = 5;            % Number of scales to run the detector
params.scale_step = 1.01;               % The scale factor

% Initialize
params.visualization = 0;
params.seq = seq;

% Run tracker
[results] = tracker(params);
