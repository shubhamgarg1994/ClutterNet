clear;

addpath(genpath('SUNRGBDtoolbox'));
addpath(genpath('tensorIO_matlab'));

ground_truth_file =  './SUNRGBDMeta.mat'; % came from above
load( ground_truth_file);
SUNRGBD = SUNRGBDMeta;

fp = fopen('./scene_data_new/test_ids_mix.txt','r');
count = fscanf(fp, '%d', 1);
TEST_IDS = fscanf(fp, '%d', count);
fclose(fp);

fp = fopen('./scene_data_new/train_ids_mix.txt','r');
count = fscanf(fp, '%d', 1);
TRAIN_IDS = fscanf(fp, '%d', count);
fclose(fp);

load('./scene_data_new/label.mat');

%% write color images as binary
batch_size = length(TRAIN_IDS);
scale = [224, 224];

% save as tensor 
images_tensor.name = 'image';
images_tensor.value = single(zeros([scale(1) scale(2) 3 batch_size]));
for ii = 1:batch_size
    fprintf('%d\n', ii);
    im = imread(SUNRGBD(TRAIN_IDS(ii)).rgbpath);
    im = imresize(im, [scale(1), scale(2)]);
    if size(im,3) == 1
        im = repmat(im, [1 1 3]);
    end
    im = single(im(:,:,[3 2 1]));
    images_tensor.value(:,:,:,ii) = single(im);
end
images_tensor.type = 'half';
images_tensor.sizeof = 2;
images_tensor.dim = 4;
writeTensors('./scene_data_new/train_image_mix.tensor', images_tensor);

batch_size = length(TEST_IDS);
images_tensor.name = 'image';
images_tensor.value = single(zeros([scale(1) scale(2) 3 batch_size]));
for ii = 1:batch_size
    fprintf('%d\n', ii);
    im = imread(SUNRGBD(TEST_IDS(ii)).rgbpath);
    im = imresize(im, [scale(1), scale(2)]);
    if size(im,3) == 1
        im = repmat(im, [1 1 3]);
    end
    im = single(im(:,:,[3 2 1]));
    images_tensor.value(:,:,:,ii) = single(im);
end
images_tensor.type = 'half';
images_tensor.sizeof = 2;
images_tensor.dim = 4;
writeTensors('./scene_data_new/test_image_mix.tensor', images_tensor);

%% write label as binary
batch_size = length(TRAIN_IDS);
label_tensor.name = 'label';
label_tensor.value = single(zeros([1, 1, 1, batch_size]));
for ii = 1:batch_size
    if ii<=length(train_label)
        label_tensor.value(:,:,:,ii) = train_label(ii);
    else
        label_tensor.value(:,:,:,ii) = 0;
    end
end
label_tensor.type = 'half';
label_tensor.sizeof = 2;
label_tensor.dim = 4;
writeTensors('./scene_data_new/train_label_mix.tensor', label_tensor);

batch_size = length(TEST_IDS);
label_tensor.name = 'label';
label_tensor.value = single(zeros([1, 1, 1, batch_size]));
for ii = 1:batch_size
    if ii<=length(test_label)
        label_tensor.value(:,:,:,ii) = test_label(ii);
    else
        label_tensor.value(:,:,:,ii) = 0;
    end
end
label_tensor.type = 'half';
label_tensor.sizeof = 2;
label_tensor.dim = 4;
writeTensors('./scene_data_new/test_label_mix.tensor', label_tensor);

%% write data file name
train_ids = TRAIN_IDS;
test_ids = TEST_IDS;

alltrain = cell(1, length(train_ids));
alltest  = cell(1, length(test_ids));
for a = 1:length(train_ids)
    alltrain{a} = ['root_to_SUNRGBD' SUNRGBDMeta(train_ids(a)).sequenceName];
end

for a = 1:length(test_ids)
    alltest{a} = ['root_to_SUNRGBD' SUNRGBDMeta(test_ids(a)).sequenceName];
end

save('./allsplit_me_full.mat', 'alltrain', 'alltest');

%% write binary depth image
SUNRGBDMeta_all = SUNRGBD;
parfor i = 1:length(SUNRGBDMeta_all)
    fprintf('%d\n', i);
    filename = sprintf('./fulldata/%06d_depth.bin', i);
    if exist(filename, 'file')
        continue;
    end
    
    data = SUNRGBDMeta_all(i);
    [rgb,points3d,depthInpaint,imsize]=read3dPoints(data);
    
    fp = fopen( filename, 'wb');
    fwrite(fp, uint32(imsize), 'uint32');
    d = depthInpaint';
    fwrite(fp, single(d(:)), 'single');
    fclose(fp);
end