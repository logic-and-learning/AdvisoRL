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

#include <stdlib.h>

#include <list>
#include <queue>
#include <string>

#include <stdio.h>

#ifdef _WIN32
#include <winsock.h>
#else
#include <arpa/inet.h>
#endif

#include "libalf/alphabet.h"
#include "libalf/normalizer.h"
#include "libalf/normalizer_msc.h"
#include "libalf/serialize.h"

namespace libalf {

using namespace std;

msc::msc_node::msc_node()
{{{
	id = -1;
	label = -1;
	process_in = NULL;
	process_out = NULL;
	buffer_in = NULL;
	buffer_out = NULL;
}}}

msc::msc_node::~msc_node()
{{{
	// delete references from other nodes
	if(process_in)
		process_in->process_out = NULL;
	if(buffer_in)
		buffer_in->buffer_out = NULL;
}}}



normalizer_msc::normalizer_msc()
{{{
	max_buffer_length = 0;
	buffers = NULL;
	buffercount = 0;
	label_bound = 0;
}}}

normalizer_msc::normalizer_msc(vector<int> &total_order, vector<int> &process_match, vector<int> &buffer_match, int max_buffer_length)
{{{
	this->total_order = total_order;
	this->process_match = process_match;
	this->buffer_match = buffer_match;
	this->max_buffer_length = max_buffer_length;

	label_bound = total_order.size();
	if(label_bound > process_match.size())
		label_bound = process_match.size();

	if(max_buffer_length > 0) {
		buffercount = 0;
		for(vector<int>::iterator vi = buffer_match.begin(); vi != buffer_match.end(); vi++)
			if(*vi > (int)buffercount)
				buffercount = *vi;

		buffercount += 1;
		buffers = new queue<int>[buffercount];

		if(label_bound > buffer_match.size())
			label_bound = buffer_match.size();
	} else {
		buffercount = 0;
		buffers = NULL;
	}

}}}

normalizer_msc::~normalizer_msc()
{{{
	if(buffers)
		delete[] buffers;
}}}

void normalizer_msc::clear()
{{{
	total_order.clear();
	process_match.clear();
	buffer_match.clear();
	max_buffer_length = 0;
	if(buffers) {
		delete[] buffers;
		buffers = NULL;
	}
	buffercount = 0;
	label_bound = 0;
}}}

basic_string<int32_t> normalizer_msc::serialize() const
{{{
	basic_string<int32_t> ret;

	ret += 0; // length field, will be filled in later.
	ret += ::serialize(normalizer::NORMALIZER_MSC);
	ret += ::serialize(total_order);
	ret += ::serialize(process_match);
	ret += ::serialize(buffer_match);
	ret += ::serialize(max_buffer_length);
	ret[0] = htonl(ret.size() - 1);

	return ret;
}}}

bool normalizer_msc::deserialize(serial_stretch & serial)
{{{
	int s;

	if(!::deserialize(s, serial)) goto deserialization_failed;
	if(s < 2) goto deserialization_failed;
	if(!::deserialize(s, serial)) goto deserialization_failed;
	if(s != normalizer::NORMALIZER_MSC) goto deserialization_failed;
	if(!::deserialize(total_order, serial)) goto deserialization_failed;
	if(!::deserialize(process_match, serial)) goto deserialization_failed;
	if(!::deserialize(buffer_match, serial)) goto deserialization_failed;

	return true;

deserialization_failed:
	clear();
	return false;
}}}

bool normalizer_msc::deserialize_extension(serial_stretch & serial)
{{{
	// the expected format:
	//
	// int normalizer::NORMALIZER_MSC
	// int count
	// {
	//	int total_order			(is appended to this->total_order)
	//	int process_match		(is appended to this->process_match)
	//	int buffer_match		(is appended to this->buffer_match)
	// } <count> times

	int count;
	int data;

	if(!::deserialize(data, serial)) goto deserialization_failed;
	if(data != normalizer::NORMALIZER_MSC) goto deserialization_failed;

	if(!::deserialize(count, serial)) goto deserialization_failed;

	total_order.reserve(total_order.size() + count);
	process_match.reserve(process_match.size() + count);
	buffer_match.reserve(buffer_match.size() + count);

	while(count) {
		if(!::deserialize(data, serial)) goto deserialization_failed;
		total_order.push_back(data);

		if(!::deserialize(data, serial)) goto deserialization_failed;
		process_match.push_back(data);

		if(!::deserialize(data, serial)) goto deserialization_failed;
		buffer_match.push_back(data);

		--count;
	}

	return true;

deserialization_failed:
	clear();
	return false;
}}}

list<int> normalizer_msc::prefix_normal_form(const list<int> & w, bool & bottom) const
{{{
	list<int> ret;
	int i;

	list<int>::const_iterator wi;
	// create MSC
	for(wi = w.begin(), i = 0; wi != w.end(); wi++, i++) {
		if(*wi < 0 || *wi >= (int)label_bound)
			goto bottom_fast;
		graph_add_node(i, *wi, true);
	}

	// MSC -> word
	while( ! graph.empty() ) {
		i = graph_reduce(true);
		if(i < 0)
			goto bottom;
		ret.push_back(i);
	}

	bottom = false;
	clear_buffers(w);
	return ret;

bottom:
	clear_buffers(w);
	ret.clear();

bottom_fast:
	graph.clear();
	bottom = true;
	ret.push_back(BOTTOM_CHAR);
	return ret;
}}}

list<int> normalizer_msc::suffix_normal_form(const list<int> & w, bool & bottom) const
{{{
	list<int> ret;
	int i;

	list<int>::const_reverse_iterator rwi;
	// create MSC (reversed)
	for(rwi = w.rbegin(), i = 0; rwi != w.rend(); rwi++, i++) {
		if(*rwi < 0 || *rwi >= (int)label_bound)
			goto bottom_fast;
		graph_add_node(i, *rwi, false);
	}

	// MSC -> word
	while( ! graph.empty() ) {
		i = graph_reduce(false);
		if(i < 0)
			goto bottom;
		ret.push_front(i);
	}

	bottom = false;
	clear_buffers(w);
	return ret;

bottom:
	clear_buffers(w);
	ret.clear();

bottom_fast:
	graph.clear();
	bottom = true;
	ret.push_back(BOTTOM_CHAR);
	return ret;
}}}




void normalizer_msc::graph_add_node(int id, int label, bool pnf) const
{{{
	msc::msc_node * newnode = new msc::msc_node();
	list<msc::msc_node*>::iterator ni, extrema;

	newnode->id = id;
	newnode->label = label;
	graph.push_back(newnode);

	// PROC-connection:
	// PNF: connection to node from other youngest node with same process that is not connected
	// SNF: connection from node to other youngest node with same process that is not connected
	extrema = graph.end();
	for(ni = graph.begin(); *ni != newnode; ni++) {
		if( (pnf && (*ni)->is_process_connected()) || (!pnf && (*ni)->is_process_referenced()) )
			continue;
		if(process_match[(*ni)->label] != process_match[label])
			continue;
		if(extrema == graph.end()) {
			extrema = ni;
		} else {
			if(pnf && (*extrema)->id < (*ni)->id)
				extrema = ni;
			if(!pnf && (*extrema)->id > (*ni)->id)
				extrema = ni;
		}
	}
	if(extrema != graph.end()) {
		if(pnf)
			(*extrema)->connect_process(newnode);
		else
			newnode->connect_process(*extrema);
	}

	// MSG-connection:
	// PNF: if this is a receiving event, connect to oldest corresponding send-event that is not connected
	// SNF: if this is a sending event, connect from oldest corresponding send-event that is not connected
	extrema = graph.end();
	if( (pnf && (total_order[label] % 2)) || (!pnf && (total_order[label] % 2 == 0)) ){
		// pnf: odd, receive-event
		// snf: even, send-event
		for(ni = graph.begin(); *ni != newnode; ni++) {
			if( ((*ni)->label / 2 != label / 2))
				continue;
			if(pnf) {
				if((total_order[(*ni)->label] % 2) || (*ni)->is_buffer_connected())
					continue;
			} else {
				if((total_order[(*ni)->label] % 2 == 0) || (*ni)->is_buffer_referenced())
					continue;
			}
			if(extrema == graph.end()) {
				extrema = ni;
			} else {
				if((*ni)->id < (*extrema)->id)
					extrema = ni;
			}
		}
	}
	if(extrema != graph.end()) {
		if(pnf)
			(*extrema)->connect_buffer(newnode);
		else
			newnode->connect_buffer(*extrema);
	}
}}}

void normalizer_msc::clear_buffers(const list<int> & word) const
// clear any buffer that this word may have touched
{{{
	if(!buffers)
		return;

	list<int>::const_iterator wi;
	int buf;

	for(wi = word.begin(); wi != word.end(); wi++) {
		buf = buffer_match[*wi];
		if(buf >= 0 && buf < (int)buffercount)
			if(!buffers[buf].empty())
				buffers[buf].pop();
	}
}}}

bool normalizer_msc::check_buffer(int label, bool pnf) const
{{{
	// check if the message `label' can be put into its buffer / taken from its buffer (true)
	// or its buffer is full / another message is at the head of the buffer (false)

	if(!buffers)
		return true;

	int buffer = buffer_match[label];

	if( (pnf && (total_order[label] % 2)) || (!pnf && (total_order[label] % 2 == 0)) ) {
		// pnf/odd: receive-event
		// snf/even: act like receive-event
		// fail if buffer is empty or a different msg is at front
		if(buffers[buffer].empty())
			return false;
		if(buffers[buffer].front() != total_order[label] / 2)
			return false;
	} else {
		// pnf/even: send-event
		// snf/odd: act like send-event
		// check buffer size
		if(max_buffer_length > 0) {
			// keep the one message in mind that is not yet in the buffer
			// (thus _equal_ or larger)
			if(buffers[buffer].size() >= (unsigned int)max_buffer_length)
				return false;
		}
	}

	return true;
}}}

void normalizer_msc::advance_buffer_status(int label, bool pnf) const
{{{
	if(buffers) {
		if( (pnf && (total_order[label] % 2)) || (!pnf && (total_order[label] % 2 == 0)) ) {
			// receive
			buffers[buffer_match[label]].pop();
		} else {
			// send
			buffers[buffer_match[label]].push(total_order[label]/2);
		}
	}
}}}

int normalizer_msc::graph_reduce(bool pnf) const
{{{
	list<msc::msc_node*>::iterator ni, extrema;

	extrema = graph.end();
	if(pnf) {
		// PREFIX normal form
		for(ni = graph.begin(); ni != graph.end(); ni++) {
			// only use minimal nodes (without incoming edges)
			if((*ni)->is_process_referenced() || (*ni)->is_buffer_referenced())
				continue;
			if(extrema == graph.end()) {
				if(check_buffer((*ni)->label, true))
					extrema = ni;
				continue;
			}
			// if there exists a minimal receive-event, never fall back to send-events
			if((total_order[(*extrema)->label] % 2) && (total_order[(*ni)->label] % 2 == 0) )
				continue;
			if((total_order[(*extrema)->label] % 2 == 0) && (total_order[(*ni)->label] % 2) ) {
				if(check_buffer((*ni)->label, true))
					extrema = ni;
				continue;
			}
			if(total_order[(*ni)->label] < total_order[(*extrema)->label]) {
				if(check_buffer((*ni)->label, true))
					extrema = ni;
			}
		}
	} else {
		// SUFFIX normal form
		for(ni = graph.begin(); ni != graph.end(); ni++) {
			// only use maximal nodes (without outgoing edges)
			if((*ni)->is_process_connected() || (*ni)->is_buffer_connected())
				continue;
			if(extrema == graph.end()) {
				if(check_buffer((*ni)->label, false))
					extrema = ni;
				continue;
			}
			// if there exists a minimal send-event, never fall back to receive-events
			if((total_order[(*extrema)->label] % 2 == 0) && (total_order[(*ni)->label] % 2) )
				continue;
			if((total_order[(*extrema)->label] % 2) && (total_order[(*ni)->label] % 2 == 0) ) {
				if(check_buffer((*ni)->label, false))
					extrema = ni;
				continue;
			}
			if(total_order[(*ni)->label] < total_order[(*extrema)->label]) {
				if(check_buffer((*ni)->label, false))
					extrema = ni;
			}
		}
	}

	if(extrema == graph.end())
		return -1;

	int label;

	label = (*extrema)->label;
	(*extrema)->disconnect();
	delete *extrema;
	graph.erase(extrema);

	advance_buffer_status(label, pnf);

	return label;
}}}

void normalizer_msc::graph_print() const
{{{
	list<msc::msc_node*>::iterator ni;

	printf("~~~ graph\n");

	for(ni = graph.begin(); ni != graph.end(); ni++) {
		printf(" node %d: label %d\n", (*ni)->id, (*ni)->label);
		printf("   process con: ");
		if((*ni)->is_process_connected())
			printf("to %d ", (*ni)->process_out->id);
		if((*ni)->is_process_referenced())
			printf("from %d", (*ni)->process_in->id);
		printf("\n" "   buffer con: ");
		if((*ni)->is_buffer_connected())
			printf("to %d", (*ni)->buffer_out->id);
		if((*ni)->is_buffer_referenced())
			printf("from %d", (*ni)->buffer_in->id);
		printf("\n");
	}

	printf("---\n");
}}}

}; // end namespace libalf

