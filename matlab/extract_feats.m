% function feat_out = extract_feats(raw)
N=size(raw,2);
z_th = 164;
s_th = 99;
zero = (2^16-1)/2;
mean_mav = repmat(mean(raw,2),[1,N]);
raw_demean = raw-mean_mav;

mav=sum(abs(raw_demean),2);

last = raw_demean(:,1:end-2);
next = raw_demean(:,2:end);

zero_change = (next.*raw_demean(:,1:end-1) < 0) & ((abs(next) >= z_th) | (abs(raw_demean(:,1:end-1))>=z_th));
zc = sum(zero_change, 2);

next_s = next(:,2:end) - raw_demean(:,2:end-1);
last_s = raw_demean(:,2:end-1) - last;
sign_change = ((next_s > 0) & (last_s < 0)) | ((next_s < 0) & (last_s > 0));
th_check = (abs(next_s) > s_th) | (abs(last_s) > (s_th));
ssc = sum(sign_change & th_check, 2);

wl = sum(abs(next - raw_demean(:,1:end-1)), 2);

feat_out = [mav,wl,zc,ssc];
feat_out = feat_out/200;
