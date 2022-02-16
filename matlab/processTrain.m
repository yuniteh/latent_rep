function [data_out, params_out] = processTrain(skip,filename)
start_dir='\\192.168.20.2'; %CAN MODIFY STARTING SEARCH DIRECTORY HERE TO MORE QUICKLY FIND DESIRED USER FOLDER

if nargin<2 || isempty(filename)
    filename = uigetdir(start_dir,'Please Select CAPS USER Folder');
    assert(any(filename~=0), 'No file was chosen.');
end

inc = 25;
win = 200;

if isempty(skip)
    [data, params] = importTrain(filename);
else
    load([filename '\DATA\MAT\traindata.mat']);
    data = data_all;
end

data_out = NaN((size(data{1},1)/inc)*length(data),size(data{1},2),win);
params_out = NaN((size(data{1},1)/inc)*length(data),2);

i = 1;
for trial=1:length(data)
    for win_i = 1:inc:size(data{trial},1)-win
        data_out(i,:,:) = data{trial}(win_i:win_i+win-1,:)';
        params_out(i,:) = params(trial,:);
    end
end

end
