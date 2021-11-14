%% Load data
filename = uigetfile();
load(filename);
ch = str2double(filename(3));
raw = data.daq.DAQ_DATA(:,ch);
t = data.daq.t;
win = data.setup.DAQ_FRAME;
inc = data.setup.DAQ_FRINC;
%% Separate into windows
raw_win = zeros(length(raw)/25,200);

ind = 1;
for i = 1:25:length(raw)-200
    raw_win(ind,:) = raw(i:i+200-1,:);
    ind = ind + 1;
end
%% plot all
figure
plot(raw)
ylim([-5,5])
%% Plot through windows on loop
for i = 1:size(raw_win,1)
    plot(raw_win(i,:))
    ylim([-5,5])
    pause
end
%% temp
raw_un = raw_win;

%% Threshold
thresh = 1;
ind_over = sum(raw_un > thresh,2);
ind_under = sum(raw_un < -thresh,2);
ind_all = ind_over + ind_under;

raw_win = raw_un(ind_all > 5,:);

%% Save windowed data
savefile = strcat('noise_win/',filename(1:end-4),'_win');

save(savefile,'raw_win')