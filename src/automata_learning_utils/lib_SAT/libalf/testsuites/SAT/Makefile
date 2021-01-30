# $Id$
# SAT testsuite Makefile

.PHONY: all clean jpegs install

include ../../config.mk
CXXFLAGS+=-Xlinker -rpath=${LIBDIR}

CPPFLAGS+=-I../../include -I${INCLUDEDIR} -Wall -ggdb3

# LDFLAGS+=-L../../src -L${LIBDIR} -lm -lalf

all: sat_static sat_regex sat_file sat_angluin_tester sat_data_file

nodata:
	-rm -f *.dot *.jpg *.bs *.txt 2>&1

clean: nodata
	-rm -f *.o sat_regex sat_static sat_file sat_angluin_tester sat_data_file 2>&1

jpegs:
	for EACH in *.dot; do dot -Tjpg "$$EACH" -o "$$EACH.jpg"; done;

sat_static: CPPFLAGS+=-Dsat_DEBUG_EQ_CLASSES
sat_static: sat_static.cpp -lm -lalf

sat_file: CPPFLAGS+=-Dsat_DEBUG_EQ_CLASSES
sat_file: sat_file.cpp -lm -lalf

sat_data_file: CPPFLAGS+=-Dsat_DEBUG_EQ_CLASSES
sat_data_file: sat_data_file.cpp -lm -lalf






