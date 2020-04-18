CC          := -gcc
CFLAGS      := -pedantic-errors -Wall -Wextra -Werror
LDFLAGS     := -L/usr/lib -lstdc++ -lm -lpthread
OBJ_DIR     := ./build
TARGET      := ./gdtomo
INCLUDE     := -Iinclude/
SRC         :=                 \
   $(wildcard src/*/*.c)       \
   $(wildcard src/*.c)         \
   $(wildcard test/*.c)     \

OBJECTS := $(SRC:%.c=$(OBJ_DIR)/%.o)

all: build $(TARGET)

$(OBJ_DIR)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<

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

