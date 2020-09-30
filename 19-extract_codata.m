sgydata = read_segy_file('Field/Data/rawfielddata/wurtsmith_line1.sgy');

rawdata = sgydata.traces;
header_details = sgydata.headers;

%imagesc(rawdata)

%extract common offset data
srcx = header_details(8,:);
srcy = header_details(9,:);
recx = header_details(11,:);
recy = header_details(12,:);

distance = sqrt((srcx-recx).^2 +(srcy-recy).^2);

%finding  indices of closest zero offset.
min_distance = min(distance); 
ZeroOffest_idx = (distance == min_distance);

%extract common offset data
co_data = rawdata(:,ZeroOffest_idx);

%cut to 400 ns
co_data_cut = co_data(1:400,:);

%save
save('Field/Data/rawfielddata/codata.mat','co_data_cut')