CC          := -gcc
NVCC        := nvcc
CFLAGS      := -pedantic-errors -Wall -Wextra -Werror
LDFLAGS     := -L/usr/lib -lstdc++ -lm -lpthread -L/usr/local/cuda/lib64 -lcudart -L/usr/local/MATLAB/R2017b/bin/glnxa64 -lmat -lmx
OBJ_DIR     := ./build
TARGET      := ./gdtomo
INCLUDE     := -Iinclude/ -I/usr/local/MATLAB/R2017b/extern/include/
SRC_C       :=                 \
   $(wildcard src/*/*/*.c)     \
   $(wildcard src/*/*.c)       \
   $(wildcard src/*.c)         \
   $(wildcard test/*.c)        \

SRC_CU      :=                 \
   $(wildcard src/*/*/*.cu)      \
   $(wildcard src/*/*.cu)      \
   $(wildcard src/*.cu)        \
   $(wildcard test/*.cu)       \

OBJECTS := $(SRC_C:%.c=$(OBJ_DIR)/%.o)
OBJECTS += $(SRC_CU:%.cu=$(OBJ_DIR)/%.o)

all: build $(TARGET)

$(OBJ_DIR)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<

$(OBJ_DIR)/%.o: %.cu
	@mkdir -p $(@D)
	$(NVCC) $(INCLUDE) -o $@ -c $<

$(TARGET): $(OBJECTS)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(OBJECTS) $(INCLUDE) $(LDFLAGS) -o $(TARGET)  

.PHONY: all build clean debug release

build:
	@mkdir -p $(OBJ_DIR)

debug: CFLAGS += -DDEBUG -g
debug: all

release: CFLAGS += -O2
release: all

edit:
	vim ./src/*.c ./src/*/*.c ./include/*/*.h ./test/test.c

clean:
	-@rm -rvf $(OBJ_DIR)
	-@rm $(TARGET)

