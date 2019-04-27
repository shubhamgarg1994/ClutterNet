load('./prepare_data/std_anchor.mat');

NUMBER_OF_MODEL = length(std_anchor);
NUMBER_OF_RANDOM = 1;

CLASS_MODELS_NAME = cell(NUMBER_OF_MODEL,1);
CLASS_MODELS_NAME{1} = 'deepScene70RL';
CLASS_MODELS_NAME{2} = 'deepScene71RL';
CLASS_MODELS_NAME{3} = 'deepScene72RL';
CLASS_MODELS_NAME{6} = 'deepScene73RL';

SNAPSHOT_NUMBER = [0 0 0 0 0 0];

exp_id = 1;

%% create file structure
curr_path = pwd;
root_dir = [curr_path '/./experiment_cmd/exp_' num2str(exp_id) '/'];
mkdir(root_dir);
for a = 1:NUMBER_OF_MODEL
    if isempty(CLASS_MODELS_NAME{a})
        continue;
    end
    scene_dir = [root_dir num2str(a) '/'];
    mkdir(scene_dir);
    for c = 1:NUMBER_OF_RANDOM
        scene_random_dir = [scene_dir num2str(c) '/'];
        mkdir(scene_random_dir);
        for b = 1:length(std_anchor(a).entity_valid)
            object_dir = [scene_random_dir num2str(b-1) '/'];
            mkdir(object_dir);
        end
    end
end


%% create network json for each scene template
AVAILABLE_GPU_ID = {'3'};
GPU_NUM = length(AVAILABLE_GPU_ID);

for a = 1:NUMBER_OF_MODEL
    if isempty(CLASS_MODELS_NAME{a})
        continue;
    end
    network_path = sprintf('./network/%s.json', CLASS_MODELS_NAME{a});
    model_string = file2string(network_path);

    template = '"shuffle_data": ';
    p1 = strfind(model_string, template);
    p2 = strfind(model_string(p1(2):end), ',') + p1(2) - 1;
    start_pos = p1(2) + length(template) - 1;
    end_pos   = p2(1);
    model_string = [model_string(1:start_pos) 'false' model_string(end_pos:end)];

    template = '"batch_size": ';
    p1 = strfind(model_string, template);
    p2 = strfind(model_string(p1(2):end), ',') + p1(2) - 1;
    start_pos = p1(2) + length(template) - 1;
    end_pos   = p2(1);
    model_string = [model_string(1:start_pos) '2' model_string(end_pos:end)];

    template = '"noise_scale": [';
    p1 = strfind(model_string, template);
    p2 = strfind(model_string(p1(2):end), ']') + p1(2) - 1;
    start_pos = p1(2) + length(template) - 1;
    end_pos   = p2(1);
    model_string = [model_string(1:start_pos) '0, 0, 0' model_string(end_pos:end)];

    for b = 1:NUMBER_OF_RANDOM
        template = '"GPU": [';
        p1 = strfind(model_string, template);
        p2 = strfind(model_string(p1(1):end), ']') + p1(1) - 1;
        start_pos = p1(1) + length(template) - 1;
        end_pos   = p2(1);
        model_string = [model_string(1:start_pos) AVAILABLE_GPU_ID{b} model_string(end_pos:end)];

        scene_dir = [root_dir num2str(a) '/'];
        fp = fopen([scene_dir 'scene_network_test_rand' AVAILABLE_GPU_ID{b} '.json'], 'w');
        fprintf(fp, '%s', model_string);
        fclose(fp);
    end

end

%% create command line to run
fp = fopen(sprintf('%s/cmd1.sh', root_dir),'w');
fprintf(fp, '#!/bin/bash\n');
fprintf(fp, 'export PATH=$PATH:/usr/local/cuda/bin\n');
fprintf(fp, 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-7.5/lib64\n');
fprintf(fp, 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cudnn/v4rc/lib64\n');

count = 0;
for a = 1:NUMBER_OF_MODEL
    if isempty(CLASS_MODELS_NAME{a})
        continue;
    end
    scene_dir = [root_dir num2str(a) '/'];

    for c = 1:NUMBER_OF_RANDOM
        fprintf(fp, 'echo ">>>>>>>>>>>> Now we start evaluating %d-th model, %d times <<<<<<<<<<<<<"\n', a, c);
        scene_random_dir = [scene_dir num2str(c) '/'];

        network_path = [scene_dir 'scene_network_test_rand' AVAILABLE_GPU_ID{c} '.json'];
        if SNAPSHOT_NUMBER(a) > 0
            model_path = sprintf('/n/fs/sunhome/DeepContext_release/detection/network/%s_snapshot_%d.marvin', strrep(CLASS_MODELS_NAME{a},'Scene','SceneV'), SNAPSHOT_NUMBER(a));
        else
            model_path = sprintf('/n/fs/sunhome/DeepContext_release/detection/network/%s.marvin', strrep(CLASS_MODELS_NAME{a},'Scene','SceneV'));
        end
        layers = cell(0,1);
        layers{1} = 'noise_tsl';
        obj_num = length(std_anchor(a).entity_valid);
        for b = 1:obj_num
            layers{end+1} = sprintf('cls_val_%02d', b-1);
            layers{end+1} = sprintf('reg_val_%02d', b-1);
            layers{end+1} = sprintf('box_%02d', b-1);
            layers{end+1} = sprintf('obj_cls_%02d', b-1);
            layers{end+1} = sprintf('fc_cat_%02d_reg', b-1);
        end
        layersName = cell(0,1);
        layersName{1} = [scene_random_dir layers{1} '.tensor'];
        for b = 1:obj_num
            object_dir = [scene_random_dir num2str(b-1) '/'];
            layersName{end+1} = [object_dir 'cls_val.tensor'];
            layersName{end+1} = [object_dir 'reg_val.tensor'];
            layersName{end+1} = [object_dir 'box.tensor'];
            layersName{end+1} = [object_dir 'obj_cls.tensor'];
            layersName{end+1} = [object_dir 'fc_cat_reg.tensor'];
        end

        count = count + 1;
        fprintf(fp, '%s &\n', generateTestCMD(network_path, model_path, layers, layersName));
        fprintf(fp, 'sleep 5\n');
        if count == 1
            fprintf(fp, 'wait\n');
            count = 0;
        end
    end

end
fclose(fp);  

%% run it here or in shell
system(sprintf('chmod 777 %s/cmd1.sh', root_dir));
system(sprintf('%s/cmd1.sh', root_dir));

