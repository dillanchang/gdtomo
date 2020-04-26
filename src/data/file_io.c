#include <data/file_io.h>

#include <data/data_types.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void read_2d_metadata(FILE* file, Data_2d* d){
  const int NUM_DIM_DIGITS = 7;

  char magic_string[6];
  fread(magic_string, 1, 6, file);
  fgetc(file); // char major_version
  fgetc(file); // char minor_version

  unsigned short header_len;
  fread(&header_len, 1, 2, file);

  char curr;
  char found_shape  = 0;
  char shape_str[6];
  char dim1_str[NUM_DIM_DIGITS];
  char dim2_str[NUM_DIM_DIGITS];
  char found_descr  = 0;
  char descr_str[6];
  char file_type_str[4];

  while(!found_shape){
    curr = fgetc(file);
    if(curr=='\''){
      fread(shape_str, 1, 5, file);
      shape_str[5] = 0;
      if(strcmp(shape_str,"shape") == 0){
        found_shape = 1;
        while(fgetc(file)!='('){;}
        for(int i=0; i<NUM_DIM_DIGITS; i++){
          dim1_str[i] = fgetc(file);
          if(dim1_str[i] == ','){
            dim1_str[i] = 0;
            i = NUM_DIM_DIGITS;
          }
        }
        fgetc(file);
        for(int i=0; i<NUM_DIM_DIGITS; i++){
          dim2_str[i] = fgetc(file);
          if(dim2_str[i] == ')'){
            dim2_str[i] = 0;
            i = NUM_DIM_DIGITS;
          }
        }
        if(fgetc(file)!=','){
          printf("%s\n","ERROR: Cannot process 2d data shape");
          exit(1);
        }
        fseek(file,10,SEEK_SET);
      }
    }
  }

  while(!found_descr){
    curr = fgetc(file);
    if(curr=='\''){
      fread(descr_str, 1, 5, file);
      descr_str[5] = 0;
      if(strcmp(descr_str,"descr") == 0){
        found_descr = 1;
        fgetc(file);
        while(fgetc(file)!='\''){;}
        fread(file_type_str, 1, 3, file);
        file_type_str[3] = 0;
        if(strcmp(file_type_str,"<f8") != 0){
          printf("%s\n","ERROR: File type not double");
          exit(1);
        }
        fseek(file,10,SEEK_SET);
      }
    }
  }

  unsigned int* dim = malloc(sizeof(unsigned int)*2);
  dim[0] = atoi(dim1_str);
  dim[1] = atoi(dim2_str);
  alloc_2d_data(d, dim);
  fseek(file,header_len+10,SEEK_SET);
}

void read_2d_data(FILE* file, Data_2d* d){
  double f;
  for(unsigned int j=0; j < (d->dim)[1]; j++){
    for(unsigned int i=0; i < (d->dim)[0]; i++){
      fread(&f, 8, 1, file);
      (d->data)[i][j] = f;
    }
  }
}

void import_2d_data(char* filename, Data_2d* d){
  FILE * file;
  file = fopen(filename, "r");
  if(file==NULL){
    printf("%s\n","ERROR: Cannot open 2d data file");
    exit(1);
  }
  read_2d_metadata(file, d);
  read_2d_data(file, d);
  fclose(file);
}

void read_3d_metadata(FILE* file, Data_3d* d){
  const int NUM_DIM_DIGITS = 7;

  char magic_string[6];
  fread(magic_string, 1, 6, file);
  fgetc(file); // char major_version
  fgetc(file); // char minor_version

  unsigned short header_len;
  fread(&header_len, 1, 2, file);

  char curr;
  char found_shape  = 0;
  char shape_str[6];
  char dim1_str[NUM_DIM_DIGITS];
  char dim2_str[NUM_DIM_DIGITS];
  char dim3_str[NUM_DIM_DIGITS];
  char found_descr  = 0;
  char descr_str[6];
  char file_type_str[4];

  while(!found_shape){
    curr = fgetc(file);
    if(curr=='\''){
      fread(shape_str, 1, 5, file);
      shape_str[5] = 0;
      if(strcmp(shape_str,"shape") == 0){
        found_shape = 1;
        while(fgetc(file)!='('){;}
        for(int i=0; i<NUM_DIM_DIGITS; i++){
          dim1_str[i] = fgetc(file);
          if(dim1_str[i] == ','){
            dim1_str[i] = 0;
            i = NUM_DIM_DIGITS;
          }
        }
        fgetc(file);
        for(int i=0; i<NUM_DIM_DIGITS; i++){
          dim2_str[i] = fgetc(file);
          if(dim2_str[i] == ','){
            dim2_str[i] = 0;
            i = NUM_DIM_DIGITS;
          }
        }
        fgetc(file);
        for(int i=0; i<NUM_DIM_DIGITS; i++){
          dim3_str[i] = fgetc(file);
          if(dim3_str[i] == ')'){
            dim3_str[i] = 0;
            i = NUM_DIM_DIGITS;
          }
        }
        if(fgetc(file)!=','){
          printf("%s\n","ERROR: Cannot process 3d data shape");
          exit(1);
        }
        fseek(file,10,SEEK_SET);
      }
    }
  }

  while(!found_descr){
    curr = fgetc(file);
    if(curr=='\''){
      fread(descr_str, 1, 5, file);
      descr_str[5] = 0;
      if(strcmp(descr_str,"descr") == 0){
        found_descr = 1;
        fgetc(file);
        while(fgetc(file)!='\''){;}
        fread(file_type_str, 1, 3, file);
        file_type_str[3] = 0;
        if(strcmp(file_type_str,"<f8") != 0){
          printf("%s\n","ERROR: File type not double");
          exit(1);
        }
        fseek(file,10,SEEK_SET);
      }
    }
  }

  unsigned int* dim = malloc(sizeof(unsigned int)*3);
  dim[0] = atoi(dim1_str);
  dim[1] = atoi(dim2_str);
  dim[2] = atoi(dim3_str);
  alloc_3d_data(d, dim);
  fseek(file,header_len+10,SEEK_SET);
}

void read_3d_data(FILE* file, Data_3d* d){
  double f;
  for(unsigned int k=0; k < (d->dim)[2]; k++){
    for(unsigned int j=0; j < (d->dim)[1]; j++){
      for(unsigned int i=0; i < (d->dim)[0]; i++){
        fread(&f, 8, 1, file);
        (d->data)[i][j][k] = f;
      }
    }
  }
}

void import_3d_data(char* filename, Data_3d* d){
  FILE * file;
  file = fopen(filename, "r");
  if(file==NULL){
    printf("%s\n","ERROR: Cannot open 3d data file");
    exit(1);
  }
  read_3d_metadata(file, d);
  read_3d_data(file, d);
  fclose(file);
}

void export_2d_data(char* filename, Data_2d* d){
  const int NUM_DIM_DIGITS = 20;
  FILE * file;

  file = fopen(filename,"w");
  if(file==NULL){
    printf("%s\n","ERROR: Cannot open 2d data file to export");
    exit(1);
  }

  int pad = 128;
  const char magic_string[] = "\x93\x4e\x55\x4d\x50\x59";
  fwrite(magic_string, 1, 6, file);       // Magic String 
  pad = pad - 6;
  fputc('\x01',file);                     // Major file number
  fputc('\x00',file);                     // Minor file number
  pad = pad - 2;
  fputc('\x76',file); fputc('\x00',file); // Header Size
  pad = pad - 2;
  fputc('\x7b',file);                     // {
  pad = pad - 1;
  const char descr[] = "\x27\x64\x65\x73\x63\x72\x27\x3a\x20\x27\x3c\x66\x38\x27\x2c\x20";
  fwrite(descr, 1, 16, file);             // 'descr': '<f8', 
  pad = pad - 16;
  const char fortran[] = "\x27\x66\x6f\x72\x74\x72\x61\x6e\x5f\x6f\x72\x64\x65\x72\x27\x3a\x20\x54\x72\x75\x65\x2c\x20";
  fwrite(fortran, 1, 23, file);           // 'fortran order': True, 
  pad = pad - 23;
  const char shape[] = "\x27\x73\x68\x61\x70\x65\x27\x3a\x20\x28";
  fwrite(shape, 1, 10, file);             // 'shape': (
  pad = pad - 10;
  char dim_buffer[NUM_DIM_DIGITS];

  sprintf(dim_buffer,"%d",(d->dim)[0]);
  char curr = dim_buffer[0];
  int idx = 0;
  while(curr != 0){
    fputc(curr,file);                     // [Dimension 1]
    pad = pad - 1;
    idx = idx + 1;
    curr = dim_buffer[idx];
  }
  fputc('\x2c',file);                     // ,
  fputc('\x20',file);                     // [space]
  pad = pad - 2;
  sprintf(dim_buffer,"%d",(d->dim)[1]);
  curr = dim_buffer[0];
  idx = 0;
  while(curr != 0){
    fputc(curr,file);                     // [Dimension 1]
    pad = pad - 1;
    idx = idx + 1;
    curr = dim_buffer[idx];
  }
  const char end_dict[] = "\x29\x2c\x20\x7d";
  fwrite(end_dict, 1, 4, file);           // ), }
  pad = pad - 4;
  for(int i=0; i < pad - 1; i++){
    fputc('\x20',file);
  }
  fputc('\x0a',file);

  double f;
  for(unsigned int j=0; j < (d->dim)[1]; j++){
    for(unsigned int i=0; i < (d->dim)[0]; i++){
      f = (d->data)[i][j];
      fwrite(&f, 8, 1, file);
    }
  }
  fputc('\x0a',file);

  fclose(file);
}

void export_3d_data(char* filename, Data_3d* d){
  const int NUM_DIM_DIGITS = 20;
  FILE * file;

  file = fopen(filename,"w");
  if(file==NULL){
    printf("%s\n","ERROR: Cannot open 3d data file to export");
    exit(1);
  }

  int pad = 128;
  const char magic_string[] = "\x93\x4e\x55\x4d\x50\x59";
  fwrite(magic_string, 1, 6, file);       // Magic String 
  pad = pad - 6;
  fputc('\x01',file);                     // Major file number
  fputc('\x00',file);                     // Minor file number
  pad = pad - 2;
  fputc('\x76',file); fputc('\x00',file); // Header Size
  pad = pad - 2;
  fputc('\x7b',file);                     // {
  pad = pad - 1;
  const char descr[] = "\x27\x64\x65\x73\x63\x72\x27\x3a\x20\x27\x3c\x66\x38\x27\x2c\x20";
  fwrite(descr, 1, 16, file);             // 'descr': '<f8', 
  pad = pad - 16;
  const char fortran[] = "\x27\x66\x6f\x72\x74\x72\x61\x6e\x5f\x6f\x72\x64\x65\x72\x27\x3a\x20\x54\x72\x75\x65\x2c\x20";
  fwrite(fortran, 1, 23, file);           // 'fortran order': True, 
  pad = pad - 23;
  const char shape[] = "\x27\x73\x68\x61\x70\x65\x27\x3a\x20\x28";
  fwrite(shape, 1, 10, file);             // 'shape': (
  pad = pad - 10;
  char dim_buffer[NUM_DIM_DIGITS];

  sprintf(dim_buffer,"%d",(d->dim)[0]);
  char curr = dim_buffer[0];
  int idx = 0;
  while(curr != 0){
    fputc(curr,file);                     // [Dimension 1]
    pad = pad - 1;
    idx = idx + 1;
    curr = dim_buffer[idx];
  }
  fputc('\x2c',file);                     // ,
  fputc('\x20',file);                     // [space]
  pad = pad - 2;
  sprintf(dim_buffer,"%d",(d->dim)[1]);
  curr = dim_buffer[0];
  idx = 0;
  while(curr != 0){
    fputc(curr,file);                     // [Dimension 2]
    pad = pad - 1;
    idx = idx + 1;
    curr = dim_buffer[idx];
  }
  fputc('\x2c',file);                     // ,
  fputc('\x20',file);                     // [space]
  pad = pad - 2;
  sprintf(dim_buffer,"%d",(d->dim)[2]);
  curr = dim_buffer[0];
  idx = 0;
  while(curr != 0){
    fputc(curr,file);                     // [Dimension 2]
    pad = pad - 1;
    idx = idx + 1;
    curr = dim_buffer[idx];
  }
  const char end_dict[] = "\x29\x2c\x20\x7d";
  fwrite(end_dict, 1, 4, file);           // ), }
  pad = pad - 4;
  for(int i=0; i < pad - 1; i++){
    fputc('\x20',file);
  }
  fputc('\x0a',file);

  double f;
  for(unsigned int k=0; k < (d->dim)[2]; k++){
    for(unsigned int j=0; j < (d->dim)[1]; j++){
      for(unsigned int i=0; i < (d->dim)[0]; i++){
        f = (d->data)[i][j][k];
        fwrite(&f, 8, 1, file);
      }
    }
  }
  fputc('\x0a',file);

  fclose(file);
}

