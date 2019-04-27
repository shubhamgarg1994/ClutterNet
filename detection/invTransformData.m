function [ objects ] = invTransformData( objects, Rot, Tsl )
%INVTRANSFORMDATA Summary of this function goes here
%   Detailed explanation goes here
invRot = inv(Rot);

for a = 1:length(objects)
    ox = objects(a).orientation(1);
    oy = objects(a).orientation(2);
    orientation(1) = invRot(1)*ox + invRot(3)*oy;
    orientation(2) = invRot(2)*ox + invRot(4)*oy;
    orientation(3) = objects(a).orientation(3);
    
    cx = objects(a).center(1);
    cy = objects(a).center(2);
    cz = objects(a).center(3);
    center(1) = invRot(1)*cx + invRot(3)*cy - Tsl(1);
    center(2) = invRot(2)*cx + invRot(4)*cy - Tsl(2);
    center(3) = cz - Tsl(3);
    
    objects(a).orientation = orientation;
    objects(a).center = center;
end

end

