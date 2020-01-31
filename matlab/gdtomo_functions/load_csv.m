function data = load_csv(filename)
  meta = fopen([filename,'_meta.txt'],'r');
  num_dim = fscanf(meta,'%d');
  shape = num_dim(2:end)';
  data = readmatrix([filename,'.csv'])';
  data = reshape(data,shape);
end
