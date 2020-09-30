%% Preprocess fdrawgathers and velocity - Part1
% removing invalid rows


%%
AllRawGathers = struct2array(load('Field/Data/fdrawgathers.mat'));
vel_td_stack = struct2array(load('Field/Data/veltd_raw.mat'))';


%%

%Check NaN
[rows, columns] = find(isnan(AllRawGathers));
rows_NaN = unique(rows);

%remove rows with NaN
AllRawGathers(rows_NaN,:) = []; %remove corresponding rows
vel_td_stack(rows_NaN,:) = [];

%Remove rows with very very large numbers, e.g. 13085
rows_Invalid=[];
for i=1:length(AllRawGathers)
   each_line = AllRawGathers(i,:);
   last_var = abs(each_line(:,end));
   if last_var > 10
       rows_Invalid =[rows_Invalid; i];
   end
end    


AllRawGathers(rows_Invalid,:) = [];
vel_td_stack(rows_Invalid,:) = [];

%% Save

save('Field/Data/AllRawGathers.mat','AllRawGathers');
save('Field/Data/veltd_raw_corr.mat','vel_td_stack');



