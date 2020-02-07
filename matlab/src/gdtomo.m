% ==============================================
% ===========  gdtomo recon driver  ============
% ==============================================

% === PARAMETERS ===============================
GDTOMO_PATH     = '/home/dillan/Projects/gdtomo/';
projs_filename  = "./data/multiple_tilt/projs.mat" ;
angles_filename = "./data/multiple_tilt/angles.mat";
recon_filename  = "./data/multiple_tilt/recon.mat" ;
err_filename    = "./data/multiple_tilt/err.mat"   ;
num_iter        = 50;
recon_alpha     = 0.5;
recon_dim       = [50,50,50];
num_cores       = 6;
% ==============================================

time_str = ['cache/t',datestr(now,'yyyymmddTHHMMSS')];
projs    = importdata(projs_filename);
angles   = importdata(angles_filename);
params   = [num_iter, recon_alpha, recon_dim(1), recon_dim(2), recon_dim(3) ...
  num_cores];

mkdir(time_str);
gen_csv(projs, ['./',time_str,'/projs' ]);
gen_csv(angles,['./',time_str,'/angles']);
gen_csv(params,['./',time_str,'/params']);

recon_info_fn = ['./',time_str,'/recon_info.txt'];
recon_info = fopen(recon_info_fn,'w');
fprintf(recon_info,'%s\n',[pwd,'/',time_str,'/projs' ]);
fprintf(recon_info,'%s\n',[pwd,'/',time_str,'/angles']);
fprintf(recon_info,'%s\n',[pwd,'/',time_str,'/params']);
fprintf(recon_info,'%s\n',[pwd,'/',time_str,'/recon' ]);
fprintf(recon_info,'%s\n',[pwd,'/',time_str,'/err'   ]);
fclose(recon_info);

system([GDTOMO_PATH, 'gdtomo recon ', recon_info_fn]);
recon = load_csv([pwd,'/',time_str,'/recon']);
err   = load_csv([pwd,'/',time_str,'/err']);
save(recon_filename,'recon');
save(err_filename,'err');
rmdir(time_str,'s');
