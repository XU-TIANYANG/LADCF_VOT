% Compile libraries and download network for LADCF
[path_root, name, ext] = fileparts(mfilename('fullpath'));

% mtimesx
if exist('external_libs/mtimesx', 'dir') == 7
    cd external_libs/mtimesx
    mtimesx_build;
    cd(path_root)
end

% PDollar toolbox
if exist('external_libs/pdollar_toolbox/external', 'dir') == 7
    cd external_libs/pdollar_toolbox/external
    toolboxCompile;
    cd(path_root)
end

% matconvnet
if exist('external_libs/matconvnet/matlab', 'dir') == 7
    cd external_libs/matconvnet/matlab
    vl_compilenn;
    cd(path_root)
    
    % donwload network
    cd feature_extraction
    mkdir networks
    cd networks
    if ~(exist('imagenet-resnet-50-dag.mat', 'file') == 2)
        disp('Downloading the network "imagenet-resnet-50-dag.mat" from "http://www.vlfeat.org/matconvnet/models/imagenet-resnet-50-dag.mat"...')
        urlwrite('http://www.vlfeat.org/matconvnet/models/imagenet-resnet-50-dag.mat', 'imagenet-resnet-50-dag.mat')
        disp('Done!')
    end
    cd(path_root)
else
    error('LADCF : Matconvnet not found.')
end