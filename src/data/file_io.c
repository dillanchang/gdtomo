#include <data/file_io.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void read_2d_metadata(char* filename, Data_2d* d){
  FILE * meta;
  const char* meta_suffix = "_meta.txt";

  // Generating file name
  unsigned int strlen_fn = strlen(filename);
  char meta_fn[strlen_fn + strlen(meta_suffix)];
  strcpy(meta_fn,filename);
  strcat(meta_fn,meta_suffix);

  // Importing meta data
  int num_dim;
  meta = fopen(meta_fn,"r");
  if(meta==NULL){
    printf("%s\n","ERROR: Cannot open meta files");
    exit(1);
  }
  fscanf(meta,"%d",&num_dim);
  if(num_dim != 2){
    printf("%s\n","ERROR: Tried to open a non 2D array in import_2d");
    exit(1);
  }
  d->dim = (unsigned int*)malloc(2*sizeof(unsigned int));
  for(int i=0; i<num_dim; i++){
    fscanf(meta,"%u",&(d->dim)[i]);
  }
  fclose(meta);
}

void read_3d_metadata(char* filename, Data_3d* d){
  FILE * meta;
  const char* meta_suffix = "_meta.txt";

  // Generating file name
  unsigned int strlen_fn = strlen(filename);
  char meta_fn[strlen_fn + strlen(meta_suffix)];
  strcpy(meta_fn,filename);
  strcat(meta_fn,meta_suffix);

  // Importing meta data
  int num_dim;
  meta = fopen(meta_fn,"r");
  if(meta==NULL){
    printf("%s\n","ERROR: Cannot open meta files");
    exit(1);
  }
  fscanf(meta,"%d",&num_dim);
  if(num_dim != 3){
    printf("%s\n","ERROR: Tried to open a non 3D array in import_3d");
    exit(1);
  }
  d->dim = (unsigned int*)malloc(3*sizeof(unsigned int));
  for(int i=0; i<num_dim; i++){
    fscanf(meta,"%u",&(d->dim)[i]);
  }
  fclose(meta);
}

void read_2d_data(char* filename, Data_2d* d){
  FILE * file;
  const char* data_suffix = ".csv";
  
  // Generating file name
  unsigned int strlen_fn = strlen(filename);
  char data_fn[strlen_fn + strlen(data_suffix)];
  strcpy(data_fn,filename);
  strcat(data_fn,data_suffix);

  // Importing data
  file = fopen(data_fn,"r");
  if(file==NULL){
    printf("%s\n","ERROR: Cannot open data file");
    exit(1);
  }

  for(unsigned int j=0; j<(d->dim)[1]; j++){
    for(unsigned int i=0; i<(d->dim)[0]; i++){
      fscanf(file,"%lf,",&(d->data)[i][j]);
    }
  }
  fclose(file);
}

void read_3d_data(char* filename, Data_3d* d){
  FILE * file;
  const char* data_suffix = ".csv";
  
  // Generating file name
  unsigned int strlen_fn = strlen(filename);
  char data_fn[strlen_fn + strlen(data_suffix)];
  strcpy(data_fn,filename);
  strcat(data_fn,data_suffix);

  // Importing data
  file = fopen(data_fn,"r");
  if(file==NULL){
    printf("%s\n","ERROR: Cannot open data file");
    exit(1);
  }

  for(unsigned int k=0; k<(d->dim)[2]; k++){
    for(unsigned int j=0; j<(d->dim)[1]; j++){
      for(unsigned int i=0; i<(d->dim)[0]; i++){
        fscanf(file,"%lf,",&(d->data)[i][j][k]);
      }
    }
  }
  fclose(file);
}

void import_2d_data(char* filename, Data_2d* d){
  read_2d_metadata(filename, d);
  alloc_2d_data(d, d->dim);
  read_2d_data(filename, d);
}

void import_3d_data(char* filename, Data_3d* d){
  read_3d_metadata(filename, d);
  alloc_3d_data(d, d->dim);
  read_3d_data(filename, d);
}

void export_2d_data(char* filename, Data_2d* d){
  FILE * meta_file;
  FILE * data_file;
  const char* meta_suffix = "_meta.txt";
  const char* data_suffix = ".csv";
  int num_dim = 2;

  // Generating file names
  unsigned int strlen_fn = strlen(filename);
  char meta_fn[strlen_fn + strlen(meta_suffix)+1];
  strcpy(meta_fn,filename);
  strcat(meta_fn,meta_suffix);
  strlen_fn = strlen(filename);
  char data_fn[strlen_fn + strlen(data_suffix)+1];
  strcpy(data_fn,filename);
  strcat(data_fn,data_suffix);

  // Exporting meta data
  meta_file = fopen(meta_fn,"w");
  if(meta_file==NULL){
    printf("%s\n","ERROR: Cannot open meta file");
    exit(1);
  }
  fprintf(meta_file,"%d\n",num_dim);
  for(int i=0; i<num_dim; i++){
    fprintf(meta_file,"%d\n",(d->dim)[i]);
  }

  // Exporting data
  data_file = fopen(data_fn,"w");
  if(data_file==NULL){
    printf("%s\n","ERROR: Cannot open data file");
    exit(1);
  }
  for(unsigned int j=0; j<(d->dim)[1]; j++){
    for(unsigned int i=0; i<(d->dim)[0]; i++){
      if(i==(d->dim)[0]-1){
        fprintf(data_file,"%0.9f",(d->data)[i][j]);
      }
      else{
        fprintf(data_file,"%0.9f,",(d->data)[i][j]);
      }
    }
    fprintf(data_file,"\n");
  }

  // Closing files
  fclose(meta_file);
  fclose(data_file);
}

void export_3d_data(char* filename, Data_3d* d){
  FILE * meta_file;
  FILE * data_file;
  const char* meta_suffix = "_meta.txt";
  const char* data_suffix = ".csv";
  int num_dim = 3;

  // Generating file names
  unsigned int strlen_fn = strlen(filename);
  char meta_fn[strlen_fn + strlen(meta_suffix)];
  strcpy(meta_fn,filename);
  strcat(meta_fn,meta_suffix);
  strlen_fn = strlen(filename);
  char data_fn[strlen_fn + strlen(data_suffix)];
  strcpy(data_fn,filename);
  strcat(data_fn,data_suffix);

  // Exporting meta data
  meta_file = fopen(meta_fn,"w");
  if(meta_file==NULL){
    printf("%s\n","ERROR: Cannot open meta file");
    exit(1);
  }
  fprintf(meta_file,"%d\n",num_dim);
  for(int i=0; i<num_dim; i++){
    fprintf(meta_file,"%d\n",(d->dim)[i]);
  }

  // Exporting data
  data_file = fopen(data_fn,"w");
  if(data_file==NULL){
    printf("%s\n","ERROR: Cannot open data file");
    exit(1);
  }
  for(unsigned int k=0; k<(d->dim)[2]; k++){
    for(unsigned int j=0; j<(d->dim)[1]; j++){
      for(unsigned int i=0; i<(d->dim)[0]; i++){
        if(i==(d->dim)[0]-1){
          fprintf(data_file,"%0.9f",(d->data)[i][j][k]);
        }
        else{
          fprintf(data_file,"%0.9f,",(d->data)[i][j][k]);
        }
      }
      fprintf(data_file,"\n");
    }
  }

  // Closing files
  fclose(meta_file);
  fclose(data_file);
}

