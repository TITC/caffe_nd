%function layers = matcaffe_extract_weights()
% layers = matcaffe_extract_weights()
% 
% Demo of how to extract network parameters ("weights") using the matlab
% wrapper.
%
%
% output
%   layers   struct array of layers and their weights
%
% You may need to do the following before you start matlab:
%  $ export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:/usr/local/cuda/lib64
%  $ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
% Or the equivalent based on where things are installed on your system

% init caffe network (spews logging info)
model_def_file='deploy.prototxt';
model_file = 'tao_iter_800.caffemodel';
net = caffe.Net(model_def_file, model_file,'test');
%net = CaffeNet.instance
layers = net.weights;
save('visualize.mat', 'layers');