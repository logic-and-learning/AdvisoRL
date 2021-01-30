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
 * (c) 2010 David R. Piegdon <david-i2@piegdon.de>
 * Author: David R. Piegdon <david-i2@piegdon.de>
 *
 */

#ifndef __libalf_algorithm_mvcl_angluinlike_h__
# define __libalf_algorithm_mvcl_angluinlike_h__

#include <list>
#include <vector>
#include <string>
#include <algorithm>
#include <functional>
#include <ostream>
#include <sstream>

#include <stdio.h>

#ifdef _WIN32
#include <winsock.h>
#else
#include <arpa/inet.h>
#endif

#include <libalf/alphabet.h>
#include <libalf/logger.h>
#include <libalf/learning_algorithm.h>

#include <libalf/triple.h>

namespace libalf {

/*
 * mVCA_angluinlike implements a L*-like algorithm to learn visible m-bounded 1-counter automata.
 * it is meant to be an implementation of the algorithm described in
 *    "D. Neider, C. Löding - Learning Visibly One-Counter Automata in Polynomial Time"
 *    Technical Report AIB-2010-02, RWTH Aachen, January 2010.
 * obtainable at
 *    http://aib.informatik.rwth-aachen.de/2010/2010-02.pdf
 *
 * NOTE: for now, this version does only support bool as <answer>.
 * ALSO NOTE:
 *       the implementation is incomplete. missing is the creation of the full equivalence query
 *       (i.e. an mVCA). everything else is implemented, but might still contain bugs.
 *       TODO are:
 *		bool find_next_valid_m()
 *		conjecture * create_full_equivalence_query()
 */

template <class answer>
class mVCA_angluinlike : public learning_algorithm<answer> {
	public: // types
		typedef std::pair<int, std::vector<answer> > fingerprint_t;

		class sample_list : public std::list<std::list<int> > {
			// list of all samples (suffixes) for a given cv.
			public: // types
				typedef typename std::list<std::list<int> >::iterator iterator;
				typedef typename std::list<std::list<int> >::const_iterator const_iterator;

				typedef typename std::list<std::list<int> >::reverse_iterator reverse_iterator;
				typedef typename std::list<std::list<int> >::const_reverse_iterator const_reverse_iterator;
			public: // methods
				bool add_sample(const std::list<int> & sample)
				{{{
					reverse_iterator li;
					for(li = this->rbegin(); li != this->rend(); ++li)
						if(*li == sample)
							return false;
					this->push_back(sample);
					return true;
				}}}

				int get_dynamic_memory_consumption() const
				{{{
					int ret = 0;
					const_iterator li;
					std::list<int>::const_iterator i;
					for(li = this->begin(); li != this->end(); ++li) {
						ret += sizeof(std::list<int>);
						for(i = li->begin(); i != li->end(); ++i)
							ret += sizeof(int);
					}

					return ret;
				}}}
		};

		class equivalence_approximation : public triple<std::list<int>, int, std::vector<answer> > {
			// a single row in a single table, containing the acceptances of prefix.suffixes,
			// where suffixes are the suffixes belonging to the current cv (see sample_list).
			public: // methods
				inline std::list<int>		& prefix()		{ return this->first; };
				inline const std::list<int>		& prefix()		const { return this->first; };

				inline int			& cv()			{ return this->second; };
				inline const int		& cv()			const { return this->second; };

				inline std::vector<answer>		& acceptances()		{ return this->third; };
				inline const std::vector<answer>	& acceptances()		const { return this->third; };

				inline equivalence_approximation()
				{ };

				inline equivalence_approximation(const std::list<int> & prefix, int cv)
				{ this->prefix() = prefix; this->cv() = cv; };

				fingerprint_t fingerprint() const
				{ return std::pair<int, std::vector<answer> >(cv(), acceptances()); }

				inline bool equivalent(equivalence_approximation & other)
				{ return cv() == other.cv() && acceptances() == other.acceptances(); }

				// check if this and the other line are equivalent under the given suffixes. i.e. they have the
				// same acceptances.
				inline bool equivalent(equivalence_approximation & other, typename sample_list::iterator suffix_it)
				{{{
					typename std::vector<answer>::iterator vi1, vi2;
					vi1 = acceptances().begin();
					vi2 = other.acceptances().begin();

					while(vi1 != acceptances().end() && vi2 != other.acceptances().end()) {
						if(*vi1 != *vi2)
							return false;

						++vi1;
						++vi2;
						++suffix_it;
					}
					return (vi1 == acceptances().end() && vi2 == other.acceptances().end());
				}}}

				// fill all missing membership information (acceptances) from knowledgebase
				// or mark them as queried
				bool fill(sample_list & samples, knowledgebase<answer> * base)
				{{{
					typename std::vector<answer>::iterator acci;
					typename sample_list::iterator sui;

					for(acci = acceptances().begin(), sui = samples.begin(); acci != acceptances().end() && sui != samples.end(); ++acci, ++sui)
						/* nothing */ ;

					if(sui == samples.end())
						return true;

					typename knowledgebase<answer>::node *pre, *word;

					pre = base->get_nodeptr(prefix());

					bool full_query = true;
					while(sui != samples.end()) {
						word = pre->find_or_create_descendant(sui->begin(), sui->end());
						if(full_query) {
							if(word->is_answered()) {
								acceptances().push_back(word->get_answer());
							} else {
								word->mark_required();
								full_query = false;
							}
						} else {
							word->mark_required();
						}
						++sui;
					}

					return full_query;
				}}}

				void print(std::ostream &os) const
				{{{
					typename std::vector<answer>::const_iterator acci;

					os << "\t\t";
					print_word(os, prefix());
					os << " (" << cv() << "):";
					for(acci = acceptances().begin(); acci != acceptances().end(); ++acci) {
						if(false == (bool)*acci)
							os << " -";
						else if(true == (bool)*acci)
							os << " +";
						else
							os << " " << *acci;
					}
					os << " ;\n";
				}}}

				int get_dynamic_memory_consumption() const
				{{{
					int ret = 0;

					typename std::list<int>::const_iterator li;
					typename std::vector<answer>::const_iterator acci;

					for(li = prefix().begin(); li != prefix().end(); ++li)
						ret += sizeof(int);

					for(acci = acceptances().begin(); acci != acceptances().end(); ++acci)
						ret += sizeof(answer);

					return ret;
				}}}
		};

		class equivalence_table : public std::list<equivalence_approximation> {
			// single table of equivalence lines for a fixed cv.
			public: // types
				typedef typename std::list<equivalence_approximation>::iterator iterator;
				typedef typename std::list<equivalence_approximation>::const_iterator const_iterator;
			public: // methods
				// find a line by its prefix
				iterator find_prefix(const std::list<int> & prefix)
				{{{
					iterator r;
					r = this->begin();
					while(r != this->end()) {
						if(r->prefix() == prefix)
							break;
						++r;
					}
					return r;
				}}}

				// find or add a line by its prefix
				iterator find_or_insert_prefix(const std::list<int> & prefix, int cv)
				{{{
					iterator r;

					r = find_prefix(prefix);
					if(r != this->end()) {
						if(r->cv() != cv)
							return this->end();
						else
							return r;
					}

					this->push_back(equivalence_approximation(prefix, cv));
					r = this->end();
					--r;
					return r;
				}}}

				// find a line by its equivalence class (same acceptances)
				iterator find_equivalence_class(equivalence_approximation & representative)
				{{{
					iterator equi;
					for(equi = this->begin(); equi != this->end(); ++equi)
						if(equi->equivalent(representative))
							break;
					return equi;
				}}}

				// fill all missing membership information (acceptances) from knowledgebase
				// or mark them as queried
				bool fill(sample_list & samples, knowledgebase<answer> * base)
				{{{
					iterator equi;
					bool complete = true;

					for(equi = this->begin(); equi != this->end(); ++equi)
						if(!equi->fill(samples, base))
							complete = false;;

					return complete;
				}}}

				void print(std::ostream & os) const
				{{{
					const_iterator it;
					for(it = this->begin(); it != this->end(); ++it)
						it->print(os);
				}}}

				int get_dynamic_memory_consumption() const
				{{{
					int ret = 0;
					const_iterator equi;
					for(equi = this->begin(); equi != this->end(); ++equi) {
						ret += sizeof(equivalence_approximation);
						ret += equi->get_dynamic_memory_consumption();
					}
					return ret;
				}}}
		};

		class m_representatives : public triple<sample_list, equivalence_table, triple<equivalence_table, equivalence_table, equivalence_table> > {
			// all information of the stratified observationtable for a fixed cv. i.e.
			// the discriminating suffixes, a table with all prefixes (state candidates)
			// and a table each for all internal (cv += 0), returning (cv -= 1) and
			// calling (cv += 1) transitions.
			public: // methods
				inline sample_list & samples()				{ return this->first; }; // suffixes
				inline const sample_list & samples()			const { return this->first; };

				inline equivalence_table & representatives()		{ return this->second; }; // prefixes
				inline const equivalence_table & representatives()	const { return this->second; };

				inline equivalence_table & returning_tr()		{ return this->third.first; }; // returning transitions
				inline const equivalence_table & returning_tr()		const { return this->third.first; };

				inline equivalence_table & internal_tr()		{ return this->third.second; }; // internal transitions
				inline const equivalence_table & internal_tr()		const { return this->third.second; };

				inline equivalence_table & calling_tr()			{ return this->third.third; }; // calling transitions
				inline const equivalence_table & calling_tr()		const { return this->third.third; };

				int get_dynamic_memory_consumption() const
				{{{
					int ret;

					ret += samples().get_dynamic_memory_consumption();

					ret += representatives().get_dynamic_memory_consumption();

					ret += returning_tr().get_dynamic_memory_consumption();
					ret += internal_tr().get_dynamic_memory_consumption();
					ret += calling_tr().get_dynamic_memory_consumption();

					return ret;
				}}}
		};

		class stratified_observationtable : public std::vector<m_representatives> {
			// full observationtable for all cv.
			public: // types
				typedef typename std::vector<m_representatives>::iterator iterator;
				typedef typename std::vector<m_representatives>::const_iterator const_iterator;

				typedef std::pair<equivalence_table *, typename equivalence_table::iterator> location;
			public: // methods
				bool fill(knowledgebase<answer> * base)
				{{{
					bool complete = true;
					iterator previous, current, next;

					// initialize iterators
					previous = this->end();
					current = this->begin();
					next = this->begin();
					if(next != this->end())
						++next;

					while(current != this->end()) {
						// fill current table:
						if(!current->representatives().fill(current->samples(), base))
							complete = false;
						// and transition-tables:
						if(previous != this->end())
							if(!current->returning_tr().fill(previous->samples(), base))
								complete = false;
						if(!current->internal_tr().fill(current->samples(), base))
							complete = false;
						if(next != this->end())
							if(!current->calling_tr().fill(next->samples(), base))
								complete = false;

						// increment iterators
						if(previous == this->end())
							previous = this->begin();
						else
							++previous;
						++current;
						if(next != this->end())
							++next;
					}

					return complete;
				}}}

				void print(std::ostream & os) const
				{{{
					int cv;
					const_iterator vi;

					for(vi = this->begin(), cv = 0; vi != this->end(); ++cv, ++vi) {
						// samples
						os << "\tcv = " << cv << ":\n";
						os << "\t  Samples:";
						typename sample_list::const_iterator si;
						for(si = vi->samples().begin(); si != vi->samples().end(); ++si) {
							os << " ";
							print_word(os, *si);
						}
						os << " ;\n";

						// representatives
						if(!vi->representatives().empty()) {
							os << "\t  Representatives:\n";
							vi->representatives().print(os);
						}

						os << "\t  Transitions:\n";
						// returning tr
						if(!vi->returning_tr().empty()) {
							os << "\t    returning (with true cv=" << cv - 1 << "):\n";
							vi->returning_tr().print(os);
						}

						// internal tr
						if(!vi->internal_tr().empty()) {
							os << "\t    internal:\n";
							vi->internal_tr().print(os);
						}

						// calling tr
						if(!vi->calling_tr().empty()) {
							os << "\t    calling (with true cv=" << cv + 1 << "):\n";
							vi->calling_tr().print(os);
						}
					}
				}}}

				int get_dynamic_memory_consumption() const
				{{{
					int ret = 0;
					const_iterator vi;

					for(vi = this->begin(); vi != this->end(); ++vi) {
						ret += sizeof(m_representatives);
						ret += vi->get_dynamic_memory_consumption();
					}

					return ret;
				}}}

				int count_words() const
				{{{
					int words = 0;

					const_iterator previous, current, next;

					// initialize iterators
					previous = this->end();
					current = this->begin();
					next = this->begin();
					if(next != this->end())
						++next;

					while(current != this->end()) {
						// current table:
						words += current->representatives().size() * current->samples().size();

						// transition-tables:
						if(previous != this->end())
							words += current->returning_tr().size() * previous->samples().size();
						words += current->internal_tr().size() * current->samples().size();
						if(next != this->end())
							words += current->calling_tr().size() * next->samples().size();

						// increment iterators
						if(previous == this->end())
							previous = this->begin();
						else
							++previous;
						++current;
						if(next != this->end())
							++next;
					}

					return words;
				}}}

				location find_representative(const std::list<int> & rep, int cv, bool & exists)
				{{{
					location ret;

					if(cv >= (int)this->size()) {
						ret.first = & ( this->operator[](0).representatives() );
						ret.second = ret.first->end();
						exists = false;
					} else {
						ret.first = & ( this->operator[](cv).representatives() );
						ret.second = ret.first->find_prefix(rep);
						exists = (ret.second != ret.first->end());
					}

					return ret;
				}}}

				location find_transition(const std::list<int> & transition, int cv_without_transition, int sigma_direction, bool & exists)
				{{{
					location ret;

					if(cv_without_transition < 0 || cv_without_transition >= (int)this->size()) {
						exists = false;
					} else {
						switch(sigma_direction) {
							case -1:
								ret.first = & ( this->operator[](cv_without_transition).returning_tr() );
								break;
							case 0:
								ret.first = & ( this->operator[](cv_without_transition).internal_tr() );
								break;
							case +1:
								ret.first = & ( this->operator[](cv_without_transition).calling_tr() );
								break;
						}
						ret.second = ret.first->find_prefix(transition);
						exists = (ret.second != ret.first->end());
					}

					if(!exists) {
						if(cv_without_transition + sigma_direction < 0 || cv_without_transition + sigma_direction >= (int)this->size()) {
							ret.first = & ( this->operator[](0).representatives() );
							ret.second = ret.first->end();
						} else {
							ret.first = & ( this->operator[](cv_without_transition + sigma_direction).representatives() );
							ret.second = ret.first->find_prefix(transition);
							exists = (ret.second != ret.first->end());
						}
					}

					return ret;
				}}}
		};

	protected: // data
		bool initialized;

		std::vector<int> pushdown_directions; // maps a label to its pushdown direction:
		// -1 == down == return ; 0 == stay == internal ; +1 == up == call

		int mode; // -1: in membership-cycle; 0: partial equivalence; 1: full equivalence;

		int known_equivalence_bound; // the m for which the current data is isomorphic to the model
		int tested_equivalence_bound; // the m for which the latest partial equivalence test was (usually known_equivalence_bound+1)
		int full_eq_test__current_m; // the current m we are testing
		int full_eq_test__queried_m; // the m that the last query was for
		std::pair<int, std::list<int> > full_eq_test__counterexample; // <height, word>

		stratified_observationtable table;

	protected: // methods
		int countervalue(std::list<int>::const_iterator word, std::list<int>::const_iterator limit, int initial_countervalue = 0)
		{{{
			for(/* nothing */; word != limit; ++word) {
				if(*word < 0 || *word >= this->get_alphabet_size()) {
					initial_countervalue = -1;
					break;
				}
				initial_countervalue += pushdown_directions[*word];
				if(initial_countervalue < 0)
					break;
			}

			return initial_countervalue;
		}}}

		int countervalue(const std::list<int> & word)
		{ return this->countervalue(word.begin(), word.end(), 0); }

		int word_height(std::list<int>::const_iterator word, std::list<int>::const_iterator limit, int initial_countervalue = 0)
		{{{
			int height = initial_countervalue;

			for(/* nothing */; word != limit; ++word) {
				if(*word < 0 || *word >= this->get_alphabet_size()) {
					initial_countervalue = -1;
					height = -1;
					break;
				}
				initial_countervalue += pushdown_directions[*word];
				if(initial_countervalue > height)
					height = initial_countervalue;
				if(initial_countervalue < 0) {
					height = -1;
					break;
				}
			}

			return height;
		}}}

		int word_height(const std::list<int> & word)
		{ return this->word_height(word.begin(), word.end(), 0); }

		typename equivalence_table::iterator insert_representative(std::list<int> rep, bool & success)
		{{{
			typename equivalence_table::iterator new_rep = table[0].representatives().end();
			int cv = countervalue(rep);

			typename equivalence_table::iterator ti;
			typename stratified_observationtable::location loc;

			if(cv < 0) {
				success = false;
				return new_rep;
			}

			// fail if it already is in the table.
			bool found;
			loc = table.find_representative(rep, cv, found);
			if(found) {
				success = false;
				return new_rep;
			}

			if(cv >= (int)table.size()) {
				// FIXME: possibly add prefixes along empty tables?
				table.resize(cv+1);
			}

			// check if there is a matching transition
			found = false;
			if(rep.size() > 0) {
				int cvd;
				cvd = pushdown_directions[rep.back()];
				loc = table.find_transition(rep, cv - cvd, cvd, found);
			}

			new_rep = table[cv].representatives().find_or_insert_prefix(rep, cv);

			if(found) {
				// if there is a matching transition, copy data from it and delete it
				new_rep->acceptances() = loc.second->acceptances();
				loc.first->erase(loc.second);
			}

			for(int sigma = 0; sigma < this->get_alphabet_size(); sigma++) {
				int ncv = cv;
				ncv += pushdown_directions[sigma];
				if(ncv < 0)
					continue;
				if(ncv > (int)table.size()) // NOTE: if == we insert so we don't loose the transition if the table is increased at some point.
					continue;
				rep.push_back(sigma);
				switch(pushdown_directions[sigma]) {
					case -1:
						table[cv].returning_tr().find_or_insert_prefix(rep, ncv);
						break;
					case 0:
						table[cv].internal_tr().find_or_insert_prefix(rep, ncv);
						break;
					case 1:
						table[cv].calling_tr().find_or_insert_prefix(rep, ncv);
						break;
				};
				rep.pop_back();
			}
			success = true;
			return new_rep;
		}}}

		bool add_partial_counterexample(std::list<int> & counterexample)
		{{{
			std::list<int> suffix;
			int cv = countervalue(counterexample);
			int height = word_height(counterexample);
			bool success;

			if(cv != 0) {
				(*this->my_logger)(LOGGER_ERROR, "mVCA_angluinlike: bad counterexample to partial equivalence query:"
								" %s has countervalue %d. should be 0.\n",
								word2string(counterexample).c_str(), cv);
				return false;
			}

			table.find_representative(counterexample, cv, success);
			if(success) {
				(*this->my_logger)(LOGGER_ERROR, "mVCA_angluinlike: counterexample to partial equivalence query"
								" %s is already contained in tables!\n",
								word2string(counterexample).c_str());
				return false;
			}

			// increase table-size so we don't incrementally do it if cv is much larger than table.size()
			if(height >= (int)table.size()) {
				// giving a counterexample that is outside the skope of this query...
				table.resize(height+1);
			}

			cv = 0;
			suffix.swap(counterexample); // we have to do this in length-increasing order, otherwise the prefix-closedness is violated

			while(!suffix.empty()) {
				insert_representative(counterexample, success);
				table[cv].samples().add_sample(suffix);

				int sigma = suffix.front();
				suffix.pop_front();
				cv += pushdown_directions[sigma];
				counterexample.push_back(sigma);
			}

			insert_representative(counterexample, success);

			mode = -1;

			return true;
		}}}

		bool add_full_counterexample(std::list<int> & counterexample)
		{{{
			int cv = countervalue(counterexample);
			int height = word_height(counterexample);
			bool success;

			if(cv != 0) {
				(*this->my_logger)(LOGGER_ERROR, "mVCA_angluinlike: bad counterexample to full equivalence query:"
								" %s has countervalue %d. should be 0.\n",
								word2string(counterexample).c_str(), cv);
				return false;
			}

			table.find_representative(counterexample, cv, success);
			if(success) {
				(*this->my_logger)(LOGGER_ERROR, "mVCA_angluinlike: counterexample to full equivalence query"
								" %s is already contained in tables!\n",
								word2string(counterexample).c_str());
				return false;
			}

			if(full_eq_test__current_m == full_eq_test__queried_m) {
				++full_eq_test__current_m;
			} else {
				(*this->my_logger)(LOGGER_WARN, "mVCA_angluinlike: You are giving multiple counterexamples to a single full equivalence query. That's ok with me, but are you sure that you know, what you are doing?\n");
			}

			// only remember this cex. if it is going out of the partial-equivalent region.
			// from these, pick the one with the smallest height.
			if(height > known_equivalence_bound && height < full_eq_test__counterexample.first) {
				full_eq_test__counterexample.first = height;
				full_eq_test__counterexample.second = counterexample;
			}

			if(full_eq_test__current_m > known_equivalence_bound) {
				// there are no more m we can test.
				// add the picked counterexample and start membership-cycle again.

				if(full_eq_test__counterexample.first < 0) {
					(*this->my_logger)(LOGGER_ERROR, "mVCA_angluinlike: you did not give a single valid counterexample to any of my full equivalence queries. (i.e. one that has a height higher than the known partial equivalence.)\n");
					--full_eq_test__current_m;
					return false;
				}

				std::list<int> suffix;

				// increase table-size so we don't incrementally do it if cv is much larger than table.size()
				if(height >= (int)table.size())
					table.resize(height+1);

				cv = 0;
				suffix.swap(counterexample); // we have to do this in length-increasing order, otherwise the prefix-closedness is violated

				while(!suffix.empty()) {
					insert_representative(counterexample, success);
					table[cv].samples().add_sample(suffix);

					int sigma = suffix.front();
					suffix.pop_front();
					cv += pushdown_directions[sigma];
					counterexample.push_back(sigma);
				}

				insert_representative(counterexample, success);

				mode = -1;

				full_eq_test__current_m = full_eq_test__queried_m = full_eq_test__counterexample.first = -1;
				full_eq_test__counterexample.second.clear();
			}

			return true;
		}}}

		virtual void initialize_table()
		{{{
			if(!initialized) {
				m_representatives mr;
				table.push_back(mr);
				std::list<int> epsilon;
				table[0].samples().add_sample(epsilon);
				bool success;
				insert_representative(epsilon, success);
				initialized = true;
			}
		}}}

		bool fill_missing_columns()
		{ return table.fill(this->my_knowledge); }

		bool close()
		// close() checks that all transitions have a corresponding equivalence class in the representatives
		{{{
			bool no_changes = true;

			typename stratified_observationtable::iterator vi;
			int cv;
			for(cv = 0, vi = table.begin(); vi != table.end(); ++vi, ++cv) {
				typename equivalence_table::iterator equi, repi;
				bool success;
				// returning transitions
				if(cv-1 >= 0) {
					for(equi = vi->returning_tr().begin(); equi != vi->returning_tr().end(); ++equi) {
						if(table[equi->cv()].representatives().find_equivalence_class(*equi) == table[equi->cv()].representatives().end()) {
							insert_representative(equi->prefix(), success)->acceptances() = equi->acceptances();
							no_changes = false;
							break;
						}
					}
				}
				// internal transitions
				{
					for(equi = vi->internal_tr().begin(); equi != vi->internal_tr().end(); ++equi) {
						if(table[equi->cv()].representatives().find_equivalence_class(*equi) == table[equi->cv()].representatives().end()) {
							insert_representative(equi->prefix(), success)->acceptances() = equi->acceptances();
							no_changes = false;
							break;
						}
					}
				}
				// calling transitions
				if(cv + 1 < (int)table.size()) {
					for(equi = vi->calling_tr().begin(); equi != vi->calling_tr().end(); ++equi) {
						if(table[equi->cv()].representatives().find_equivalence_class(*equi) == table[equi->cv()].representatives().end()) {
							insert_representative(equi->prefix(), success)->acceptances() = equi->acceptances();
							no_changes = false;
							break;
						}
					}
				}
			}

			return no_changes;
		}}}

		bool make_consistent()
		// make_consistent() checks that all transitions of two equivalent representatives a, b are equivalent as well
		{{{
			bool no_changes = true;

			typename stratified_observationtable::iterator vi;
			for(vi = table.begin(); vi != table.end(); ++vi) {
				typename equivalence_table::iterator a, b;
				for(a = vi->representatives().begin(); a != vi->representatives().end(); ++a) {
					b = a;
					++b;
					while(b != vi->representatives().end()) {
						if(a->equivalent(*b)) {
							std::list<int> wa, wb;
							wa = a->prefix();
							wb = b->prefix();
							// check that all transitions are equivalent as well
							for(int sigma = 0; sigma < this->get_alphabet_size(); ++sigma) {
								int cvd, ncv;
								cvd = pushdown_directions[sigma];
								ncv = a->cv() + cvd;
								if(ncv < 0 || ncv >= (int)table.size())
									continue;

								wa.push_back(sigma);
								wb.push_back(sigma);

								typename stratified_observationtable::location as, bs;

								bool success;
								as = table.find_transition(wa, a->cv(), cvd, success);
								bs = table.find_transition(wb, b->cv(), cvd, success);

								typename sample_list::iterator bad_suffix;
								bad_suffix = table[ncv].samples().begin();
								if(!as.second->equivalent(*bs.second, bad_suffix)) {
									std::list<int> new_suffix;
									new_suffix = *bad_suffix;
									new_suffix.push_front(sigma);
									table[ncv].samples().push_back(new_suffix);
									no_changes = false;
								}

								wa.pop_back();
								wb.pop_back();
							}
						}
						++b;
					}
				}
			}

			return no_changes;
		}}}

		virtual bool complete()
		{{{
			if(!initialized)
				initialize_table();

			if(!fill_missing_columns()) {
				mode = -1;
				return false;
			}

			(*this->my_logger)(LOGGER_ALGORITHM, "%s", this->to_string().c_str());

			if(!close()) {
				(*this->my_logger)(LOGGER_ALGORITHM, "closing...\n");
				return complete();
			}

			if(!make_consistent()) {
				(*this->my_logger)(LOGGER_ALGORITHM, "making consistent...\n");
				return complete();
			}

			if(mode == -1)
				mode = 0;

			return true;
		}}}

		conjecture * create_partial_equivalence_query()
		{{{
			bounded_simple_mVCA * cj = new bounded_simple_mVCA;

			cj->valid = true;
			cj->is_deterministic = true;
			cj->input_alphabet_size = this->get_alphabet_size();
			cj->state_count = 0; // done on the fly

			if(tested_equivalence_bound == known_equivalence_bound)
				++tested_equivalence_bound;

			cj->m_bound = tested_equivalence_bound;

			std::map<fingerprint_t, int> states; // this is not really good. something better anyone?

			// generate statemap and mark initial and final states
			typename stratified_observationtable::iterator vi;
			typename equivalence_table::iterator equi;

			for(vi = table.begin(); vi != table.end(); ++vi) {
				for(equi = vi->representatives().begin(); equi != vi->representatives().end(); ++equi) {
					fingerprint_t fingerprint = equi->fingerprint();
					if(states.find(fingerprint) == states.end()) {
						states[fingerprint] = cj->state_count;
						if(equi->prefix().empty())
							cj->initial_states.insert(cj->state_count);
						if((fingerprint.first == 0) && (true == (bool)(fingerprint.second[0])))
							cj->output_mapping[cj->state_count] = true;
						++cj->state_count;

					}

				}
			}

			// list all transitions
			for(vi = table.begin(); vi != table.end(); ++vi) {
				for(equi = vi->representatives().begin(); equi != vi->representatives().end(); ++equi) {

					fingerprint_t fequi = equi->fingerprint();
					std::list<int> rep;
					rep = equi->prefix();

					for(int sigma = 0; sigma < this->get_alphabet_size(); ++sigma) {
						int dcv = pushdown_directions[sigma];
						int ncv = equi->cv() + dcv;
						if(ncv < 0 || ncv >= (int)table.size())
							continue;

						rep.push_back(sigma);

						bool found; // (ignored)
						fingerprint_t fnew;
						fnew = table.find_transition(rep, equi->cv(), dcv, found).second->fingerprint();

						cj->transitions[ states[fequi] ][ sigma ].insert( states[fnew] );

						rep.pop_back();
					}
				}
			}

			return cj;
		}}}

		bool find_next_valid_m()
		{
#error TODO
			
			return false;
		}

		conjecture * create_full_equivalence_query()
		{
#error TODO
			// FIXME
			simple_mVCA * cj = new simple_mVCA;

			cj->valid = true;
			cj->is_deterministic = true;
			cj->input_alphabet_size = this->get_alphabet_size();
			cj->alphabet_directions = pushdown_directions;
			cj->initial_states.insert(0);
//			cj->state_count =   
//			cj->output_mapping   
//			cj->m_bound =   
//			cj->transitions   

			if(find_next_valid_m()) {
				// if there is one, create mVCA
				
			} else {
				// try the whole BG as an automaton without repeating structure
				
			}

			delete cj;
			return NULL;
			// FIXME
			return cj;
		}

		virtual conjecture * derive_conjecture()
		{{{
			switch(mode) {
				default:
				case -1:
					(*this->my_logger)(LOGGER_ERROR, "mVCA_angluinlike: bad mode %d in derive_conjecture()! (INTERNAL ERROR)\n", mode);
					return NULL;
				case 0:
					return create_partial_equivalence_query();
				case 1:
					return create_full_equivalence_query();
			}
		}}}

	public: // methods
		mVCA_angluinlike()
		{{{
			this->set_logger(NULL);
			this->set_knowledge_source(NULL);
			clear();
		}}}

		mVCA_angluinlike(knowledgebase<answer> *base, logger *log, int alphabet_size)
		{{{
			this->set_logger(log);
			this->set_knowledge_source(base);
			clear();
			this->set_alphabet_size(alphabet_size);
		}}}

		virtual enum learning_algorithm_type get_type() const
		{ return ALG_MVCA_ANGLUINLIKE; };

		virtual enum learning_algorithm_type get_basic_compatible_type() const
		{ return ALG_MVCA_ANGLUINLIKE; };

		void clear()
		{{{
			initialized = false;
			this->set_alphabet_size(0);
			mode = -1;
			known_equivalence_bound = -1;
			tested_equivalence_bound = -1;
			full_eq_test__current_m = -1;
			full_eq_test__queried_m = -1;
			full_eq_test__counterexample.first = -1;
			full_eq_test__counterexample.second.clear();
			pushdown_directions.clear();
			table.clear();
		}}}

		virtual void indicate_pushdown_alphabet_directions(const std::vector<int> & directions)
		{ pushdown_directions = directions; }

		virtual void increase_alphabet_size(int new_asize)
		{ this->set_alphabet_size(new_asize); }

		virtual void generate_statistics(void)
		{{{
			statistics["initialized"] = initialized;
			if(initialized) {
				int words = 0; // words that are stored in the tables
				int bytes = 0; // bytes this algorithm consumes over all

				bytes = sizeof(*this);
				bytes += table.get_dynamic_memory_consumption();
				words = table.count_words();

				this->statistics["table_bound"] = (int)table.size() - 1;
				this->statistics["known_equivalence_bound"] = known_equivalence_bound;
				this->statistics["size.table.words"] = words;
				this->statistics["memory.bytes"] = bytes;
			}
		}}}

		virtual bool sync_to_knowledgebase()
		{{{
			(*this->my_logger)(LOGGER_ERROR, "mVCA_angluinlike does not support sync-operation.\n");
			return false;
		}}}
		virtual bool supports_sync() const
		{ return false; };

		virtual std::basic_string<int32_t> serialize() const
		{{{
			std::basic_string<int32_t> ret;

			ret += 0; // size, filled in later.
			ret += ::serialize((int)ALG_MVCA_ANGLUINLIKE);
			ret += ::serialize(initialized);
			ret += ::serialize(this->get_alphabet_size());
			ret += ::serialize(pushdown_directions);
			ret += ::serialize(mode);
			ret += ::serialize(known_equivalence_bound);
			ret += ::serialize(tested_equivalence_bound);
			ret += ::serialize(full_eq_test__current_m);
			ret += ::serialize(full_eq_test__queried_m);
			ret += ::serialize(full_eq_test__counterexample);
			ret += ::serialize(table);

			ret[0] = htonl(ret.length() - 1);

			return ret;
		}}}
		virtual bool deserialize(serial_stretch & serial)
		{{{
			int size;
			int type;

			clear();

			if(!::deserialize(size, serial)) goto deserialization_failed;
			// total size: we don't care.
			if(!::deserialize(type, serial)) goto deserialization_failed;
			if(type != (int)ALG_MVCA_ANGLUINLIKE)
				goto deserialization_failed;
			if(!::deserialize(initialized, serial)) goto deserialization_failed;
			if(!::deserialize(size, serial)) goto deserialization_failed;
			this->set_alphabet_size(size);
			if(!::deserialize(pushdown_directions, serial)) goto deserialization_failed;
			if(!::deserialize(mode, serial)) goto deserialization_failed;
			if(!::deserialize(known_equivalence_bound, serial)) goto deserialization_failed;
			if(!::deserialize(tested_equivalence_bound, serial)) goto deserialization_failed;
			if(!::deserialize(full_eq_test__current_m, serial)) goto deserialization_failed;
			if(!::deserialize(full_eq_test__queried_m, serial)) goto deserialization_failed;
			if(!::deserialize(full_eq_test__counterexample, serial)) goto deserialization_failed;
			if(!::deserialize(table, serial)) goto deserialization_failed;

			return true;
deserialization_failed:
			clear();
			return false;
		}}}

		bool deserialize_magic(serial_stretch & serial, std::basic_string<int32_t> & result)
		{{{
			int command;

			result.clear();

			if(!::deserialize(command, serial)) return false;

			switch(command) {
				case 0: { // indicate pushdown property of alphabet
						std::vector<int> dirs;
						if(!::deserialize(dirs, serial)) return false;
						pushdown_directions = dirs;
						return true;
					}
				case 1: { // indicate partial equivalence
						result += ::serialize(indicate_partial_equivalence());
						return true;
					}
			};
			// bad command?
			return false;
		}}}

		virtual void print(std::ostream &os) const
		{{{
			os << "stratified_observationtable {\n";
			os << "\tmode: " << (     (mode == -1) ? "membership query cycle\n"
						: (mode == 0) ? "partial equivalence query cycle\n"
						: (mode == 1) ? "full equivalence query cycle\n"
						:               "UNKNOWN MODE\n"  );
			os << "\tknown equivalence bound: " << known_equivalence_bound << "\n";

			if(mode == 0 && tested_equivalence_bound != known_equivalence_bound)
					os << "\ttested equivalence bound: " << tested_equivalence_bound << "\n";

			if(mode == 1) {
				os << "\tcurrent m for full eq. query: " << full_eq_test__current_m << "\n";
				os << "\tqueried m for full eq. query: " << full_eq_test__queried_m << "\n";
				os << "\tsaved counterexample: ";
				if(full_eq_test__counterexample.first < 0) {
					os << "none yet.\n";
				} else {
					print_word(os, full_eq_test__counterexample.second);
					os << " (cv=" << full_eq_test__counterexample.first << ")\n";
				}
			}

			os << "\n\ttable data:\n";
			table.print(os);

			os << "};\n";
		}}}

		virtual bool conjecture_ready()
		{ return complete(); };

		virtual bool indicate_partial_equivalence()
		{{{
			if(mode != 0) {
				(*this->my_logger)(LOGGER_ERROR, "mVCA_angluinlike: you indicated a partial equivalence, but i did not expect that now.\n");
				return false;
			}

			known_equivalence_bound = tested_equivalence_bound;
			full_eq_test__current_m = -1;
			full_eq_test__queried_m = -1;
			full_eq_test__counterexample.first = -1;
			full_eq_test__counterexample.second.clear();
			mode = 1;

			return true;
		}}}

		virtual bool add_counterexample(std::list<int> counterexample)
		{{{
			switch(mode) {
				default:
				case -1:
					(*this->my_logger)(LOGGER_WARN, "mVCA_angluinlike: you are giving a counterexample but i did not expect one. please complete filling the table until i send you the next equivalence query.\n");
					return false;
				case 0:
					return add_partial_counterexample(counterexample);
				case 1:
					return add_full_counterexample(counterexample);
			}
		}}}
};


}; // end of namespace libalf

#endif

