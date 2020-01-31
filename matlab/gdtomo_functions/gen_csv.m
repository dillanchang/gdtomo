function gen_csv(data,filename)
  num_dim = size(size(data),2);
  d1 = size(data,1);
  d2 = 1;
  for d_idx = 2:num_dim
    d2 = d2*size(data,d_idx);
  end
  data_new = reshape(data,d1,d2)';
  writematrix(data_new,[filename,'.csv']);
  meta = fopen([filename,'_meta.txt'],'w');
  fprintf(meta,'%d\n',num_dim);
  for i=1:num_dim
    fprintf(meta,'%d\n',size(data,i));
  end
  fclose(meta);
end
