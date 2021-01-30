/*
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
 * (c) 2008,2009,2010 Lehrstuhl Softwaremodellierung und Verifikation (I2), RWTH Aachen University
 *                and Lehrstuhl Logik und Theorie diskreter Systeme (I7), RWTH Aachen University
 * (c) 2012 Daniel Neider Lehrstuhl, Logik und Theorie diskreter Systeme (I7), RWTH Aachen University
 *
 * Author: Daniel Neider (neider@automata.rwth-aachen.de),  Florian Richter
 *
 */

#ifndef __libalf_algorithm_counterexample_learning_h__
#define __libalf_algorithm_counterexample_learning_h__

// Standard includes 
#include <iostream>
#include <list>
#include <set>
#include <map>

#include <assert.h>

// libALF includes
#include "libalf/knowledgebase.h"
#include "libalf/conjecture.h"
#include "libalf/algorithm_automata_inferring.h"

namespace libalf {

/**
 * This class implements a counter-example guided learner as presented in
 * - Martin Leucker, Daniel Neider: Learning Minimal Deterministic Automata from
 *   Inexperienced Teachers. ISoLA 2012: 524-538
 *
 * Note that bool is the only answer type supported.
 */
class counterexample_learning : public learning_algorithm<bool> {

	private:

	/**
	 * Stores a pointer to the last conjecture (or NULL) a new conjecture can be derived
	 */
	finite_automaton * last_conjecture;
	
	/**
	 * Stores a reference to the RMC inferring algorithm
	 */
	automata_inferring<bool> * inferring_algorithm;
	
	public:

	/**
	 * Creates a new learning algorithm.
	 */
	counterexample_learning(knowledgebase<bool> * base, logger * log, int alphabet_size, automata_inferring<bool> * inferring_algorithm) {

		this->set_alphabet_size(alphabet_size);
		this->set_logger(log);
		this->set_knowledge_source(base);
		
		this->inferring_algorithm = inferring_algorithm;
		last_conjecture = NULL;
		
	}

	~counterexample_learning() {
	
		if(last_conjecture != NULL) {
			delete last_conjecture;
			last_conjecture = NULL;
		}
	
	}

	void set_inferring_algorithm(automata_inferring<bool> * inferring_algorithm) {
		if(inferring_algorithm == NULL)
			(*this->my_logger)(LOGGER_WARN, "Not a valid inferring algorithm.\n");
		this->inferring_algorithm = inferring_algorithm;
	}

	automata_inferring<bool> * get_inferring_algorithm() {
		return inferring_algorithm;
	}
	
	void print(std::ostream & os) const {
	
		os << "Counter-example guided learning using an automata inferring algorithm. ";
		os << "Alphabet size is " << this->alphabet_size << ".";
		
	}

	bool conjecture_ready() {

		// Check knowledgebase
		if(this->my_knowledge == NULL) {
			return false;
		}
		
		// Check whether there is an old conjecture. If so, no counter-example has been given
		if(last_conjecture == NULL) {
			return true;
		} else {
			return false;
		}
		
	}
	
	/**
	 * Adds a counter-example to the knowledgebase. The algorithm itself does
	 * not have internal data about words and their classification.
	 * 
	 * @param counter_example The counter-example to add
	 *
	 * @return Returns true if the operation was successful or false if the
	 *         algorithm does not await a counter-example.
	 */
	bool add_counterexample(std::list<int> counter_example) {
	
		// Check
		if(this->last_conjecture == NULL) {
			(*this->my_logger)(LOGGER_WARN, "Algorithm is not awaiting any counter-example.\n");
			return false;
		}
		if(this->my_knowledge == NULL) {
			(*this->my_logger)(LOGGER_WARN, "No knowledgebase given.\n");
			return false;
		}
		
		// Compute classification of the counter-example wrt. the last conjecture
		bool classification = last_conjecture->contains(counter_example);
		
		// Add counter-example to knowledgebase
		this->my_knowledge->add_knowledge(counter_example, !classification);
		
		// Delete last conjecture
		delete this->last_conjecture;
		this->last_conjecture = NULL;
		
		return true;
		
	}
	
	/**
	 * We are always ready
	 */
	bool complete() {

		return true;

	}

	/**
	 * Derive a conjecture that is consistent with the database and inductive
	 */
	conjecture * derive_conjecture() {

		// Check knowledgebase
		if(this->my_knowledge == NULL) {
			(*this->my_logger)(LOGGER_WARN, "No knowledgebase given.\n");
			return false;
		}

		// Check inferring algorithm
		if(this->inferring_algorithm == NULL) {
			(*this->my_logger)(LOGGER_WARN, "No inferring algorithm given.\n");
			return false;
		}
	
		// Check whether we should wait for a counter-example
		if(last_conjecture != NULL) {
		
			(*this->my_logger)(LOGGER_WARN, "Algorithm is waiting for a counter-example. Continuing anyway.\n");
			delete last_conjecture;
			last_conjecture = NULL;
			
		}
	
		// Prepare inferring algorithm
		inferring_algorithm->set_alphabet_size(this->alphabet_size);
		inferring_algorithm->set_knowledge_source(this->my_knowledge);
		inferring_algorithm->set_logger(this->my_logger);

		// Derive conjecture
		last_conjecture = dynamic_cast<finite_automaton *>(inferring_algorithm->advance());
		assert(last_conjecture != NULL);
		
		return new finite_automaton(*last_conjecture);
		
	}

	public:

	virtual enum learning_algorithm_type get_type() const
	{ return ALG_COUNTEREXAMPLE_LEARNING; };

	virtual enum learning_algorithm_type get_basic_compatible_type() const
	{ return ALG_COUNTEREXAMPLE_LEARNING; };

	void increase_alphabet_size(int new_alphabet_size) {
		this->alphabet_size = new_alphabet_size;
	}

	// Not implemented
	bool sync_to_knowledgebase() {
		(*this->my_logger)(LOGGER_WARN, "This feature is not implemented.\n");
		return false;
	}

	// Not implemented
	bool supports_sync() const {
		(*this->my_logger)(LOGGER_WARN, "This feature is not implemented.\n");
		return false;
	}

	std::basic_string<int32_t> serialize() const {

		std::basic_string<int32_t> ret;

		// length (filled in later)
		ret += 0;

		// implementation type
		ret += htonl(ALG_COUNTEREXAMPLE_LEARNING);

		// alphabet size
		ret += htonl(this->get_alphabet_size());

		// last conjecture
		if(last_conjecture == NULL) {
			ret += htonl(0);
		} else {
			ret += htonl(1);
			ret += last_conjecture->serialize();
		}

		ret[0] = htonl(ret.length() - 1);

		return ret;
	}

	bool deserialize(serial_stretch & serial) {
		int s;
		
		//size
		if(!::deserialize(s, serial)) return deserialization_failed();

		//check implementation type
		if(!::deserialize(s, serial)) return deserialization_failed();
		if(s != ALG_COUNTEREXAMPLE_LEARNING) return deserialization_failed();

		//alphabet size
		if(!::deserialize(s, serial)) return deserialization_failed();
		this->set_alphabet_size(s);

		//last conjecture
		int is_valid;
		::deserialize(is_valid, serial);
		if(is_valid == 1) {
			last_conjecture = new finite_automaton();
			if(!last_conjecture->deserialize(serial)) return deserialization_failed();
		}

		return true;
	}

	bool deserialization_failed() {
		this->set_alphabet_size(0);
		inferring_algorithm = NULL;
		last_conjecture = NULL;
		return false;
	}

	// Not implemented
	bool deserialize_magic(serial_stretch & serial, std::basic_string<int32_t> & result) {
		//nothing to do, all cmds handled already
		return true;
	}

};

}; // End libalf namespace

#endif
