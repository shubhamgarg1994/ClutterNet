function [ scoreMatrix ] = bb3dOverlap( bb1, bb2 )
%BB3DOVERLAP Summary of this function goes here
%   Detailed explanation goes here

nBb1 = size(bb1,2);
nBb2 = size(bb2,2);

volume1 = cuboidVolume(bb1);
volume2 = cuboidVolume(bb2);
intersection = cuboidIntersectionVolume(bb1,bb2);
union = repmat(volume1',1,nBb2)+repmat(volume2,nBb1,1)-intersection;
scoreMatrix = intersection ./ union;
        
end

