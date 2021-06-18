%% branch

fid = py.open('latent_rep\python\models_2\svae_feat_dim_3_TR_results.p','rb');
data = py.pickle.load(fid);

results = cell(9,1);
for i = 1:3
    temp = double(py.array.array('d',py.numpy.nditer(data(i))));
    temp = transpose(reshape(temp,[4,6]));
    results{i} = temp(:,[3,4,2,1]);
end

for i = 4:6
    temp = double(py.array.array('d',py.numpy.nditer(data(i))));
    results{i} = temp([3,4,2,1]);
end

for i = 1:3
    results{i+6} = std(results{i})/sqrt(size(results{i},1));
end

%%

mat = nan(4,4);
p_vals = cell(1,3);

for met = 1:3
    p_vals{met} = mat;
    for i = 1:4
        for j = 1:4
            [h,p_vals{met}(i,j)] = ttest(results{met}(:,i),results{met}(:,j));
        end
    end
    p_vals{met} = p_vals{met}*6;
end

%%
p_vals = nan(4,1);

for i = 1:4
    [h,p_vals(i)] = ttest(results{2}(:,i),results{3}(:,i));
end

%%
[h,p] = ttest(results{2}(:,1),results{3}(:,3))
