CC = g++
SRCS_TRACKING = src/tracking_class.cpp
SRCS_FORCE = src/force_estimation.cpp

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    CFLAGS = -g -Wall -std=c++11 -O3 -Wall -fPIC -shared -std=c++11 `python3 -m pybind11 --includes` -I ./
    PROG_TRACKING = src/lib/find_marker.so
    PROG_FORCE = src/lib/force_estimation.so
    ALL_PROGS = $(PROG_TRACKING) $(PROG_FORCE)
endif

ifeq ($(UNAME_S),Darwin)
    CFLAGS = -g -Wall -std=c++11 -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` -I ./
    PROG_TRACKING = src/lib/find_marker`python3-config --extension-suffix`
    PROG_FORCE = src/lib/force_estimation`python3-config --extension-suffix`
    ALL_PROGS = $(PROG_TRACKING) $(PROG_FORCE)
endif

all: $(ALL_PROGS)

$(PROG_TRACKING): $(SRCS_TRACKING)
	mkdir -p src/lib
	$(CC) $(CFLAGS) -o $(PROG_TRACKING) $(SRCS_TRACKING)

$(PROG_FORCE): $(SRCS_FORCE)
	mkdir -p src/lib
	$(CC) $(CFLAGS) -o $(PROG_FORCE) $(SRCS_FORCE)


clean:
	rm -rf src/lib