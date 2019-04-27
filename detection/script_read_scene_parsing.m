addpath('../code/tensorIO_matlab/');

scene_cls_res_file = '../scene_template_classification/score_full.tensor';
ground_truth_file = '../code/SUNRGBDMeta.mat';
mean_anchor_file = './prepare_data/mean_anchor.mat';
std_anchor_file  = './prepare_data/std_anchor.mat';

exp_id = 1;
exp_append = '/1/';
save_append = '/test/';

ground_matrix_file = cell(6,1);
ground_matrix_file{1} = '../transformation/translation_network_scene_cls/est_matrix_final.mat';
ground_matrix_file{2} = '../transformation/translation_network_scene_cls/est_matrix_final.mat';
ground_matrix_file{3} = '../transformation/translation_network_scene_cls/est_matrix_final.mat';
ground_matrix_file{6} = '../transformation/translation_network_scene_cls/est_matrix_final.mat';

TEST_SCENE_CLS = [1 2 3 6];

root_dir = ['./experiment_cmd/exp_' num2str(exp_id) '/'];

%% read test ids
load(ground_truth_file);
SUNRGBD = SUNRGBDMeta; % not used during testing
load(mean_anchor_file);
load(std_anchor_file);

fp = fopen('../code/scene_data_new/test_ids_mix.txt','r');
count = fscanf(fp, '%d', 1);
TEST_IDS = fscanf(fp, '%d', count);
fclose(fp);

ALL2USE  = false(length(SUNRGBD),1);
ALL2TEST = false(length(SUNRGBD),1);
TEST2USE = false(length(TEST_IDS),1);

ALL2TEST(TEST_IDS) = true;
ALL2USE(TEST_IDS) = true;

SUNRGBD_TEST = SUNRGBD(ALL2TEST);
TEST2USE = true(length(TEST_IDS),1);

SUNRGBD_TESTUSE = SUNRGBD_TEST( TEST2USE);

TESTUSE_IDS = TEST_IDS( TEST2USE);

TEST2USE_IDS = find(TEST2USE);

%% read estimated transformation
all_matrix = cell(6,1);
all_MATRIX_TEST = cell(6,1);
all_MATRIX_TESTUSE = cell(6,1);
for a = 1:6
    if isempty(ground_matrix_file{a})
        continue;
    end
    load(ground_matrix_file{a});
    all_matrix{a} = matrix;
    all_MATRIX_TEST{a} = all_matrix{a}(TEST_IDS);
    all_MATRIX_TESTUSE{a} = all_MATRIX_TEST{a}(TEST2USE);
end

b = cell(0,1);
for a = 1:length(TEST_SCENE_CLS)
    scn_cls = TEST_SCENE_CLS(a);
    b = [b std_anchor(scn_cls).classnames];
end
OBJECT_NAME = unique(b);

%% read scene template classification results
a = readTensor(scene_cls_res_file);
b = squeeze(a.value);
SCENE_CLS_SCORE  = b(TEST_SCENE_CLS, TEST2USE);
[~,SCENE_CLS_RESULT] = max(SCENE_CLS_SCORE, [], 1);

%% read scene parsing results
clear WALL_PRED WALL_INIT
WALL_PRED.ceiling_valid = zeros(length(TEST_IDS), 1);
WALL_PRED.floor_valid = zeros(length(TEST_IDS), 1);
WALL_PRED.wall_valid = zeros(length(TEST_IDS), 1);
WALL_PRED.ceiling_value = zeros(length(TEST_IDS), 1);
WALL_PRED.floor_value = zeros(length(TEST_IDS), 1);
WALL_PRED.wall_value = zeros(length(TEST_IDS), 2);

WALL_INIT.ceiling_valid = zeros(length(TEST_IDS), 1);
WALL_INIT.floor_valid = zeros(length(TEST_IDS), 1);
WALL_INIT.wall_valid = zeros(length(TEST_IDS), 1);
WALL_INIT.ceiling_value = zeros(length(TEST_IDS), 1);
WALL_INIT.floor_value = zeros(length(TEST_IDS), 1);
WALL_INIT.wall_value = zeros(length(TEST_IDS), 2);

PREDICTION = repmat(struct('PRED',zeros(12,0),'INIT',zeros(12,0),'count',0,'classname',[]), length(OBJECT_NAME), 1);
for a = 1:length(OBJECT_NAME)
    PREDICTION(a).classname = OBJECT_NAME{a};
end

for a = 1:length(TEST_SCENE_CLS)
    scn_cls = TEST_SCENE_CLS(a);
    anchor = std_anchor(scn_cls);
    
    noise_file = [root_dir num2str(scn_cls) exp_append 'noise_tsl.tensor'];
    noise_tensor = readTensorHyper(noise_file);
    nos_out = squeeze(noise_tensor.value);
    
    unify_trans = zeros(3, length(TEST_IDS));
    for b = 1:length(TEST_IDS)
        Rot = all_MATRIX_TEST{scn_cls}(b).Rot;
        Tsl1 = all_MATRIX_TEST{scn_cls}(b).Tsl;
        Tsl2 = all_MATRIX_TEST{scn_cls}(b).Tsl + nos_out(:,b);
        Tsl(1:2) = Rot * ( Tsl1(1:2) - Tsl2(1:2) );
        Tsl(3)   = Tsl1(3) - Tsl2(3);
        unify_trans(:,b) = Tsl;
    end
    
    for b = length(anchor.walls)+1:length(anchor.entity_valid)
        c = ismember(OBJECT_NAME, anchor.classnames{anchor.objects(b-length(anchor.walls)).type});
        d = find(c); % id in object list
        root = [root_dir num2str(scn_cls) exp_append];
        if scn_cls == 6 
            P = readResultFromDisk( root, b-1, anchor, mean_anchor(scn_cls), [0.05 0.05 0.05], [-3.2 -3.2 -1.6]);
        else
            P = readResultFromDisk( root, b-1, anchor, mean_anchor(scn_cls), [0.05 0.05 0.05], [-3.2 -2.2 -1.6]);
        end
        
        for c = 1:length(TESTUSE_IDS)
            if SCENE_CLS_RESULT(c) ~= a
                continue;
            end

            k = TEST2USE_IDS(c);
            Rot = all_MATRIX_TESTUSE{scn_cls}(c).Rot;
            Tsl = all_MATRIX_TESTUSE{scn_cls}(c).Tsl + nos_out(:, k);
            
            
            count = PREDICTION(d).count;
            count = count + 1;
            corners_pred = getBoxBoundaryR(invTransformData(P.pred_value(k), Rot, Tsl));
            corners_init = getBoxBoundaryR(invTransformData(P.init_value(k), Rot, Tsl));
            
            PREDICTION(d).PRED(:,count) = [corners_pred; k; P.cls_pre(2,k)];
            PREDICTION(d).INIT(:,count) = [corners_init; k; P.cls_pre(2,k)];
            
            PREDICTION(d).count = count;
        end
    end    
end

%% save results
save_root = ['./experiment/exp' num2str(exp_id) save_append];
mkdir(save_root);
save(sprintf('%swall.mat', save_root), 'WALL_PRED', 'WALL_INIT');
for a = 1:length(OBJECT_NAME)
    PRED = PREDICTION(a).PRED;
    INIT = PREDICTION(a).INIT;
    save(sprintf('%s%s.mat', save_root, PREDICTION(a).classname), 'PRED', 'INIT');
end




