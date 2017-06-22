% Matlab code

h5Name = 'EC2_blocks_1_8_9_15_76_89_105_CV_HG_align_window_-0.5_to_0.79_file_nobaseline.h5';
LMDBName = 'testSet';

X = h5read(h5Name, '/Xhigh gamma');
y = h5read(h5Name, '/y');
% size(X) -> 258 86 2572
% size(y) -> 2572 1
X = permute(X, [1, 2, 4, 3]); % add dimension
y = y'; % transpose

% Randomly shuffle data order.
nSamples = size(X, 3);
perm = randperm(nSamples);
X = X(:,:,perm);
y = y(perm);

X = single(X); % cast from uint8 to float
y = int32(y);

if ~exist(LMDBName, 'dir')
    mkdir(LMDBName);
end
write_lmdb(LMDBName, X, y);
