#import pdb
def evaluate_dnf(formula,true_props):
    """
    Evaluates 'formula' assuming 'true_props' are the only true propositions and the rest are false. 
    e.g. evaluate_dnf("a&b|!c&d","d") returns True 
    """
    # ORs
    if "|" in formula:
        for f in formula.split("|"):
            if evaluate_dnf(f,true_props):
                return True
        return False
    # ANDs
    if "&" in formula:
        for f in formula.split("&"):
            if not evaluate_dnf(f,true_props):
                return False
        return True
    # NOT
    if formula.startswith("!"):
        return not evaluate_dnf(formula[1:],true_props)

    # Base cases
    if formula == "True":  return True
    if formula == "False": return False
    return formula in true_props

def are_these_machines_equivalent(rm1, u1, rm2, u2):
    """
    Returns True iff this reward machine rm1 starting from u1
    is equivalent to the reward machine rm2 starting from u2
    """
    # First, we compute the subset of states that are reachable from the initial state
    U1 = _get_reachable_states(rm1, u1)
    U2 = _get_reachable_states(rm2, u2)
    #print("U1", U1)
    #print("U2", U2)
    if len(U1) != len(U2):
        # if they have different number of nodes, then they are not equivalent
        #print("False! Different number of nodes!")
        return False
    # Second, we group nodes by their degree
    D1 = _get_degrees(rm1, U1)
    D2 = _get_degrees(rm2, U2)
    #print("D1", D1)
    #print("D2", D2)
    for d in D1:
        if d not in D2 or len(D1[d]) != len(D2[d]):
            # if there degree of the nodes does not match, they are not equivalent
            #print("False! Node degrees does not match!")
            return False
    # Finally, we just consider all the possible matchings and check if the edges and rewards match
    matches = _get_all_possible_matchings(D1, D2)

    #print("matches", matches)
    for match in matches:
        # check if this match is correct
        if _check_match(rm1, rm2, match, u1, u2):
            #print("Found a match. For rm1 = {}, rm2 = {}, match is {}".format(str(rm1), str(rm2), str(match)))

            return True

    # No match actually worked out        
    #print("False! no match actually worked :/")
    return False

def _check_match(rm1, rm2, match, u_fixed_1=None, u_fixed_2=None):


    match = dict(match)

    #make sure that starting points do match
    if not match[u_fixed_1] == u_fixed_2:
        return False


    for u1 in match:
        for u2 in rm1.delta_u[u1]:
            if u2 in match:
                # checking if this edge is in the other reward machine
                m1, m2 = match[u1], match[u2]
                if m1 not in rm2.delta_u or m2 not in rm2.delta_u[m1]:
                    #print("No match! delta_u does not exists")
                    return False
                if m1 not in rm2.delta_r or m2 not in rm2.delta_r[m1]:
                    #print("No match! delta_r does not exists")
                    return False
                # checking that the reward matches
                reward1 = rm1.delta_r[u1][u2]
                reward2 = rm2.delta_r[m1][m2]
                if not reward1.compare_to(reward2):
                    #print("No match! rewards does not match")
                    return False
                # checking that the label matches
                label1 = rm1.delta_u[u1][u2]
                label2 = rm2.delta_u[m1][m2]
                if not _are_formulas_equivalent(label1, label2):
                    #print("No match! formula does not match:", label1, "=/=", label2)
                    return False
    return True

def _are_formulas_equivalent(f1, f2):
    """
    Assuming f1 and f2 are in DNF, we do a simple syntax matching to check if f1 is equivalent to f2
    (note that this procedure is sound, but not complete)
    """
    f1 = _break_DNF_formula(f1)
    f2 = _break_DNF_formula(f2)

    #print("f1",f1)
    #print("f2",f2)

    if len(f1) != len(f2):
        return False
    
    # Syntactic matching
    return _are_disjunctions_equivalent(f1,f2)

def _are_disjunctions_equivalent(d1,d2):
    if len(d1) == 0:
        return True
    for i in range(len(d1)):
        di = d1[:i] + d1[i+1:]
        for j in range(len(d2)):
            dj = d2[:j] + d2[j+1:]
            if _are_conjunctions_equivalent(d1[i],d2[j]) and _are_disjunctions_equivalent(di,dj):
                return True
    return False

def _are_conjunctions_equivalent(c1,c2):
    c12 = [c for c in c1 if c not in c2] # c1 -> c2
    c21 = [c for c in c2 if c not in c1] # c2 -> c1
    #print("c12 + c21", c12 + c21)
    return len(c12 + c21) == 0


def _break_DNF_formula(d):
    ret = []
    for c in d.split("|"):
        ret.append(set(c.split("&")))
    return ret


def _get_all_possible_matchings(D1, D2, degrees=None):
    if degrees is None:
        degrees = list(D1.keys())
    
    # base case
    if len(degrees) == 0:
        return [[]]
    # recursion
    ret = []
    if len(degrees) > 0:
        d = degrees[0]
        matches_d = _get_matching_between_sets(D1[d],D2[d])
        matches_next = _get_all_possible_matchings(D1, D2, degrees[1:])
        for md in matches_d:
            for mn in matches_next:
                ret.append(md + mn)
    return ret

def _get_matching_between_sets(D1,D2):
    # base case
    if len(D1) == 0:
        return [[]]
    # recursion
    ret = []
    for i in range(len(D1)):
        for j in range(len(D2)):
            for o in _get_matching_between_sets(D1[0:i] + D1[i+1:],D2[0:j] + D2[j+1:]):
                ret.append([(D1[i],D2[j])] + o)
    return ret


def _get_degrees(rm, U):
    degrees = {}
    for u in U:
        if u in rm.delta_u:
            degree = len(rm.delta_u[u])
            if degree not in degrees:
                degrees[degree] = []
            degrees[degree].append(u)
    return degrees

def _get_reachable_states(rm, u):
    U = set()
    _get_reachable_states_DFS(rm, u, U)
    return U

def _get_reachable_states_DFS(rm, u1, U):
    if u1 not in U:
        U.add(u1)
        if u1 in rm.delta_u:
            for u2 in rm.delta_u[u1]:
                _get_reachable_states_DFS(rm, u2, U)