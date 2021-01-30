# $Id$
# RPNI testsuite Makefile

.PHONY: all clean jpegs install

include ../../config.mk
CXXFLAGS+=-Xlinker -rpath=${LIBDIR}

CPPFLAGS+=-I../../include -I${INCLUDEDIR} -Wall -ggdb3

# LDFLAGS+=-L../../src -L${LIBDIR} -lm -lalf

all: rpni_static rpni_regex rpni_file rpni_angluin_tester rpni_data_file

nodata:
	-rm -f *.dot *.jpg *.bs *.txt 2>&1

clean: nodata
	-rm -f *.o rpni_regex rpni_static rpni_file rpni_angluin_tester rpni_data_file 2>&1

jpegs:
	for EACH in *.dot; do dot -Tjpg "$$EACH" -o "$$EACH.jpg"; done;

rpni_static: CPPFLAGS+=-DRPNI_DEBUG_EQ_CLASSES
rpni_static: rpni_static.cpp -lm -lalf

rpni_file: CPPFLAGS+=-DRPNI_DEBUG_EQ_CLASSES
rpni_file: rpni_file.cpp -lm -lalf

rpni_data_file: CPPFLAGS+=-DRPNI_DEBUG_EQ_CLASSES
rpni_data_file: rpni_data_file.cpp -lm -lalf






