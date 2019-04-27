function saveMatrix( matrix, filename )
%SAVEMATRIX Summary of this function goes here
%   Detailed explanation goes here
fp = fopen(filename, 'wb');
fwrite( fp, uint32(length(matrix)), 'uint32');
for i = 1:length(matrix)
    fwrite(fp, single(matrix(i).Rot), 'single');
    fwrite(fp, single(matrix(i).Tsl), 'single');
    fwrite(fp, single(matrix(i).R), 'single');
    fwrite(fp, single(matrix(i).K), 'single');
    fwrite(fp, uint32(matrix(i).h), 'uint32');  
    fwrite(fp, uint32(matrix(i).w), 'uint32');  
end
fclose(fp);
end

