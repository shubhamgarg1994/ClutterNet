class_def_file = './scene_cls.mat';
load( class_def_file);

fp = fopen('../code/scene_data_new/test_ids_mix.txt','r');
count = fscanf(fp, '%d', 1);
TEST_IDS = fscanf(fp, '%d', count);
fclose(fp);

scene_cls_res_file = '../scene_template_classification/score_full.tensor';
a = readTensor(scene_cls_res_file);
b = squeeze(a.value);
[B, I] = max( b([1 2 3 6],:),[],1);
conf_ids = find(B>0.99);

test_ids = 1:length(B);
exp_id = 1;
exp_append = '/test/';

figure;

for a = 1:length(cls)
    local_test_ids = test_ids;
    
    classname = cls{a};
    
    % load ground truth
    gnd_file = sprintf('./groundtruth_full/%s.mat', classname);   
    load(gnd_file);
    I = ismember( GND(11,:), local_test_ids);
    GND_loc = GND(:,I);

    % load deep sliding shape result
    bas_file = sprintf('./baseline_full/%s.mat', classname);
    load(bas_file);
    I = ismember( DET(11,:), local_test_ids);
    PRED_loc = DET(:,I);
    
    % load our result
    pre_file = sprintf('./experiment/exp%d/%s/%s.mat', exp_id, exp_append, classname);
    load(pre_file);
    I = ismember( PRED(11,:), conf_ids);
    PRED_loc = [PRED_loc PRED(:,I)];

    % do non-maximum supression
    keep_all = false(size(PRED_loc,2),1);
    for b = 1:length(local_test_ids)
        aaa = find(PRED_loc(11,:)==local_test_ids(b));
        keep = nms3d(PRED_loc(1:10,aaa), PRED_loc(12,aaa), 0.1);
        keep_all(aaa(keep)) = true;
    end
    PRED_loc = PRED_loc(:,keep_all);
    
    % evaluation: ap and pr curve
    pred_label = zeros(size(PRED_loc,2), 1);
    pred_keep  = true(size(PRED_loc,2), 1);
    gndt_label = zeros(size(GND_loc, 2), 1);
    gndt_hitby = zeros(size(GND_loc, 2), 1);
    for b = 1:length(local_test_ids)
        pred_ids = find(PRED_loc(11,:) == local_test_ids(b));
        gndt_ids = find(GND_loc(11,:) == local_test_ids(b));
        
        pred_bb = PRED_loc(1:10,pred_ids);
        pred_sc = PRED_loc(12,pred_ids);
        gnds_bb = GND_loc(1:10,gndt_ids);
        
        available_gndid = true(length(gndt_ids),1);
        available_preid = true(length(pred_ids),1);
        available_hitby = zeros(length(gndt_ids),1);
        pred_kp = true(length(pred_ids), 1);
        
        nBb1 = size(pred_bb,2);
        nBb2 = size(gnds_bb,2);

        volume1 = cuboidVolume(pred_bb);
        volume2 = cuboidVolume(gnds_bb);
        intersection = cuboidIntersectionVolume(pred_bb,gnds_bb);
        union = repmat(volume1',1,nBb2)+repmat(volume2,nBb1,1)-intersection;
        scoreMatrix = intersection ./ union;
        
        [~, seqid] = sort(pred_sc,'descend');
        for c = 1:length(seqid)
            [m,n] = max(scoreMatrix(seqid(c),:));
            if m > 0.25
                available_preid(seqid(c)) = false;
                available_gndid(n) = false;
                available_hitby(n) = pred_ids(seqid(c));
                scoreMatrix(:,n) = 0;
            end
        end
              
        pred_label(pred_ids) = ~available_preid;
        gndt_label(gndt_ids) = ~available_gndid;
        gndt_hitby(gndt_ids) =  available_hitby;    
        pred_keep(pred_ids) = pred_kp;
    end
    
    pred_label = pred_label(pred_keep);
    final_scrs = PRED_loc(12,pred_keep);
    new_pred_id = find(pred_keep);
    
    [~, det_rank] = sort(final_scrs, 'descend');
    precision = zeros(length(final_scrs), 1);
    recall    = zeros(length(final_scrs), 1);
    threshold = zeros(length(final_scrs), 1);
    for i = 1:length(final_scrs)
        threshold(i) = final_scrs(det_rank(i));
        precision(i) = sum(pred_label(det_rank(1:i))) / i;
        I = ismember(new_pred_id(det_rank(1:i)), gndt_hitby);
        recall(i) = sum(I) / length(gndt_label);
    end
    
    ap = xVOCap(recall, precision);
    
    PRECISION{a} = precision;
    RECALL{a} = recall;
    AP{a} = ap;
    
    % plot figure
    subplot(4,4,a);
    plot(recall, precision);
    legend(sprintf('AP: %4.3f', ap));
    title(cls{a}, 'Interpreter','none');
    xlim([0 1]); ylim([0 1]);
end

for a = 1:length(cls)
    fprintf('%s: %f\n', cls{a}, AP{a});
end
