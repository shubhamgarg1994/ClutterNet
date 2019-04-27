function [ output ] = readResultFromDisk( root, obj_id, std_anchor, mean_anchor, resolution, offset )
%READRESUTLFROMDISK Summary of this function goes here
%   obj_id: universal id, including wall, start from 0
cls_pre_file = fullfile( root, num2str(obj_id), 'obj_cls.tensor');
cls_gnd_file = fullfile( root, num2str(obj_id), 'cls_val.tensor');
reg_pre_file = fullfile( root, num2str(obj_id), 'fc_cat_reg.tensor');
reg_gnd_file = fullfile( root, num2str(obj_id), 'reg_val.tensor');
box_file     = fullfile( root, num2str(obj_id), 'box.tensor');

cls_pre_tensor = readTensorHyper(cls_pre_file);
cls_gnd_tensor = readTensorHyper(cls_gnd_file);
cls_pre = squeeze(cls_pre_tensor.value);
cls_gnd = squeeze(cls_gnd_tensor.value);

reg_pre_tensor = readTensorHyper(reg_pre_file);
reg_gnd_tensor = readTensorHyper(reg_gnd_file);
box_tensor = readTensorHyper(box_file);
reg_pre = squeeze(reg_pre_tensor.value);
reg_gnd = squeeze(reg_gnd_tensor.value);
box_out = squeeze(box_tensor.value);
num_box = size(box_out,2);

wall_id = 0;
object_id = 0;
is_wall = obj_id < length(std_anchor.walls);
if is_wall
    wall_id = obj_id + 1;
else
    object_id = obj_id - length(std_anchor.walls) + 1;
end

if is_wall
    init_pos = zeros(2, num_box);
    pred_pos = zeros(2, num_box);
    
    if std_anchor.walls(wall_id).type == 0
        init_pos(2,:) = (box_out(4,:) + 5) * resolution(2) + offset(2);
        pred_pos(2,:) = init_pos(2,:) - reg_pre(2,:);
    else
        init_pos(1,:) = (box_out(2,:) + 5) * resolution(3) + offset(3); 
        pred_pos(1,:) = init_pos(1,:) - reg_pre';
    end
    
else
    init_boxes = box_out(2:7,:);
    init_boxes(1,:) = init_boxes(1,:)*resolution(3) + offset(3);
    init_boxes(2,:) = init_boxes(2,:)*resolution(3) + offset(3);
    init_boxes(3,:) = init_boxes(3,:)*resolution(2) + offset(2);
    init_boxes(4,:) = init_boxes(4,:)*resolution(2) + offset(2);
    init_boxes(5,:) = init_boxes(5,:)*resolution(1) + offset(1);
    init_boxes(6,:) = init_boxes(6,:)*resolution(1) + offset(1); 
 
    init_object = repmat(struct('center',[],'size',[],'orientation',[]), num_box, 1);
    pred_object = repmat(struct('center',[],'size',[],'orientation',[]), num_box, 1);

    for j = 1:num_box
        ori = mean_anchor.objects(object_id).orientation;
        x_center = (init_boxes(5,j) + init_boxes(6,j))/2;
        y_center = (init_boxes(3,j) + init_boxes(4,j))/2;
        z_center = (init_boxes(1,j) + init_boxes(2,j))/2;
        if abs(ori(2)) > abs(ori(1))
            x_size = (init_boxes(6,j) - init_boxes(5,j))/2;
            y_size = (init_boxes(4,j) - init_boxes(3,j))/2;      
        else
            x_size = (init_boxes(4,j) - init_boxes(3,j))/2;
            y_size = (init_boxes(6,j) - init_boxes(5,j))/2;   
        end
        z_size   = (init_boxes(2,j) - init_boxes(1,j))/2;
        init_object(j).center = [x_center y_center z_center];
        init_object(j).size   = [x_size y_size z_size];
        init_object(j).orientation = mean_anchor.objects(object_id).orientation;
        
        
        x_center = x_center - reg_pre(1,j)*(std_anchor.objects(object_id).center(1)+0.01);
        y_center = y_center - reg_pre(2,j)*(std_anchor.objects(object_id).center(2)+0.01);
        z_center = z_center - reg_pre(3,j)*(std_anchor.objects(object_id).center(3)+0.01);
        x_size   = x_size - reg_pre(4,j)*(std_anchor.objects(object_id).size(1)+0.01);
        y_size   = y_size - reg_pre(5,j)*(std_anchor.objects(object_id).size(2)+0.01);
        z_size   = z_size - reg_pre(6,j)*(std_anchor.objects(object_id).size(3)+0.01);

        pred_object(j).center = [x_center y_center z_center];
        pred_object(j).size   = [x_size y_size z_size];
        pred_object(j).orientation = mean_anchor.objects(object_id).orientation;
    end
    
end

output.is_wall = is_wall;
output.wall_id = wall_id;
output.object_id = object_id;
output.cls_pre = cls_pre;
output.cls_gnd = cls_gnd;

if is_wall
    output.init_value = init_pos;
    output.pred_value = pred_pos;
else
    output.init_value = init_object;
    output.pred_value = pred_object;
end

end

