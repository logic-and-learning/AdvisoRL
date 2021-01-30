/* $Id$
 * vim: fdm=marker
 *
 * This file is part of libalf.
 *
 * libalf is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libalf is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with libalf.  If not, see <http://www.gnu.org/licenses/>.
 *
 * (c) 2008,2009 Lehrstuhl Softwaremodellierung und Verifikation (I2), RWTH Aachen University
 *           and Lehrstuhl Logik und Theorie diskreter Systeme (I7), RWTH Aachen University
 * Author: David R. Piegdon <david-i2@piegdon.de>
 *
 */

#include <ostream>
#include <fstream>
#include <string>

#ifdef _WIN32
#include <winsock.h>
#include <stdint.h>
#include <stdio.h>
#else
#include <arpa/inet.h>
#endif

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "libalf/basic_string.h"

namespace libalf {

using namespace std;

void print_basic_string(basic_string<int32_t> str, ostream &os)
{{{
	basic_string<int32_t>::iterator si;

	os << "_";

	for(si = str.begin(); si != str.end(); si++) {
		os << *si;
		os << "_";
	}
}}}

void print_basic_string_2hl(basic_string<int32_t> str, ostream &os)
{{{
	basic_string<int32_t>::iterator si;

	os << "_";

	for(si = str.begin(); si != str.end(); si++) {
		os << ntohl(*si);
		os << "_";
	}
}}}

bool basic_string_to_fd(basic_string<int32_t> &str, int fd)
{{{
	basic_string<int32_t>::iterator si;

	if(fd < 0)
		return false;

	for(si = str.begin(); si != str.end(); si++) {
		if(write(fd, &*si, sizeof(int32_t)) != sizeof(int32_t))
			return false;
	}

	return true;
}}}

bool fd_to_basic_string(const int fd, basic_string<int32_t> & str)
{{{
	if(fd < 0)
		return false;

	str.clear();
	int32_t d;
	int s;
	while((s = read(fd, &d, sizeof(d))) == sizeof(d)) {
		str.push_back(d);
	}
	if(s > 0)
		return false;

	return true;
}}}

bool basic_string_to_file(basic_string<int32_t> &str, const char* filename)
{{{
	int fd;

	fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);

	bool ok = basic_string_to_fd(str, fd);

	if(close(fd) >= 0)
		return ok;
	else
		return false;
}}}

bool file_to_basic_string(const char* filename, basic_string<int32_t> &str)
{{{
	int fd;
	str.clear();

	fd = open(filename, O_RDONLY);

	bool ok = fd_to_basic_string(fd, str);

	if(close(fd) >= 0)
		return ok;
	else
		return false;
}}}

std::ostream & operator<<(std::ostream& os, const basic_string<int32_t> &str)
// str[0] has to be the size of the string (otherwise operator>> won't work)
// the content of str has to be in network byte order.
{{{
	basic_string<int32_t>::const_iterator si;

	for(si = str.begin(); si != str.end(); ++si) {
		os.write((const char*)&(*si), sizeof(*si));
		if(!os.good())
			break;
	}

	return os;
}}}

std::istream & operator>>(std::istream& is, basic_string<int32_t> &str)
// the string in <is> has to be in network byte order, and the first element
// has to be the size of the full string.
{{{
	int32_t i, size;

	is.read((char*)&i, sizeof(i));
	if(!is.good())
		return is;
	size = ntohl(i);

	str.clear();
	str.reserve(size);

	str += i;
	for( /* nil */ ; size > 0; --size) {
		is.read((char*)&i, sizeof(i));
		if(!is.good())
			break;
		str += i;
	}

	return is;
}}}

}; // end of namespace libalf

