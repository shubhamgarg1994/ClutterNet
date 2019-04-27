function [ bound ] = getBoxBoundaryR( boxes )
%GET Summary of this function goes here
%   Detailed explanation goes here
num_box = length(boxes);
bound = zeros(10, num_box);
for i = 1:length(num_box)
    box = boxes(i);
    direct1 = box.orientation ./ norm(box.orientation);
    direct2 = zeros(1,2);
    direct2(1) =  direct1(2);
    direct2(2) = -direct1(1);
    
    bound(1:2,i) = (box.center(1:2) - box.size(2) * direct1(1:2) - box.size(1) * direct2(1:2))';
    bound(3:4,i) = (box.center(1:2) + box.size(2) * direct1(1:2) - box.size(1) * direct2(1:2))';
    bound(5:6,i) = (box.center(1:2) + box.size(2) * direct1(1:2) + box.size(1) * direct2(1:2))';
    bound(7:8,i) = (box.center(1:2) - box.size(2) * direct1(1:2) + box.size(1) * direct2(1:2))';
    bound(9,i)   = box.center(3) - box.size(3);
    bound(10,i)  = box.center(3) + box.size(3);
end

end

