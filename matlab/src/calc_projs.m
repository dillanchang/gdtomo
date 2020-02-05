% ==============================================
% =======  gdtomo calc_projs driver  ===========
% ==============================================

% === PARAMETERS ===============================
GDTOMO_PATH = '/home/dillan/Projects/gdtomo/';
vol_filename    = "./data/test_calc_projs/vol.mat" ;
angles_filename = "./data/test_calc_projs/angles.mat";
projs_filename  = "./data/test_calc_projs/projs.mat" ;
% ==============================================

time_str = ['t',datestr(now,'yyyymmddTHHMMSS')];
vol      = importdata(vol_filename);
angles   = importdata(angles_filename);

mkdir(time_str);
gen_csv(vol,   ['./',time_str,'/vol' ]);
gen_csv(angles,['./',time_str,'/angles']);

projs_info_fn = ['./',time_str,'/projs_info.txt'];
projs_info = fopen(projs_info_fn,'w');
fprintf(projs_info,'%s\n',[pwd,'/',time_str,'/vol'   ]);
fprintf(projs_info,'%s\n',[pwd,'/',time_str,'/angles']);
fprintf(projs_info,'%s\n',[pwd,'/',time_str,'/projs' ]);
fclose(projs_info);

system([GDTOMO_PATH, 'gdtomo calc_projs ', projs_info_fn]);
proj  = load_csv([pwd,'/',time_str,'/projs']);
save(projs_filename,'proj');
rmdir(time_str,'s');
