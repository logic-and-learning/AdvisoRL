# $Id$
# libalf Makefile

.PHONY: install-all-deps all testsuites clean install uninstall

all:
	+make -C src all
	+make -C include install
	+make -C src install	
	+make -C testsuites/SAT sat_data_file
	


testsuites: install
	make -C testsuites

clean:
	+make -C src clean
	+make -C testsuites clean

install:
	+make -C include install
	+make -C src install


uninstall:
	make -C include uninstall
	make -C src uninstall

