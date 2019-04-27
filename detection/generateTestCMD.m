function [ cmd ] = generateTestCMD( network, model, layers, layersName )
%GENERATETESTCMD Summary of this function goes here
%   Detailed explanation goes here
cmd = ['./marvin test ' network ' ' model ' '];
for i = 1:length(layers)
    cmd = [cmd layers{i}];
    if i ~= length(layers)
        cmd = [cmd ','];
    else
        cmd = [cmd ' '];
    end
end

for i = 1:length(layersName)
    cmd = [cmd layersName{i}];
    if i ~= length(layersName)
        cmd = [cmd ','];
    end
end

end

