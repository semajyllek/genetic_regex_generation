
from networkx.drawing.nx_agraph import graphviz_layout
from deap import base, creator, tools, gp
from collections import defaultdict
import matplotlib.pyplot as plt
from regex_classes import *
from scoop import futures
import numpy as np
import networkx
import random
import regex
import json
import time
import sys




class RegexGenerator:
    def __init__(self, filename,
                 pop_size=200,
                 threshold=0.8,
                 max_ex=50,
                 max_regex_len=80,
                 ngen=500,
                 max_tree_ht=7,
                 min_tree_ht=3,
                 term_ratio=0.7,
                 CXPB=0.8,
                 MUTPB=0.1):

        self.PATTERN1 = regex.compile('\w+|\s+|[^\w\s]+')
        self.PATTERN2 = regex.compile('\w+|\s+|[^\w\s]')
        self.pop_size = pop_size
        self.threshold = threshold
        self.max_ex = max_ex
        self.max_regex_len = max_regex_len
        self.ngen = ngen
        self.pop_size = pop_size
        self.max_ht = max_tree_ht
        self.min_ht = min_tree_ht
        self.term_ratio = term_ratio
        self.CXPB = CXPB
        self.MUTPB = MUTPB
        self.fit_cache = defaultdict(tuple)


        # data is dict of examples, r target is the (an) regex target for the data, terms are most frequent sequences
        # num_examples and char_counts are for computing stats, fitnesses, etc.
        self.data, self.regex_target, self.terminals, self.num_examples, self.char_counts = self.get_data(filename,
                                                                                                          threshold_pct=threshold,
                                                                                                          max_ex=max_ex)
        generics = ['\w', '[A-Za-z]', '\d', '\s', '\w+', '[A-Za-z]+', '\d+', '\s+', '.*']

        self.terminals = generics + [self.escape_char(t) for t in self.terminals if t not in generics]

        # classes are extremely simple python objects with 1 string member
        self.classes = [TermClass, QClass, ChClass, NChClass, PlusClass, StarClass,
                        LookBehindClass, LookAheadClass]

        self.pset = gp.PrimitiveSetTyped(name="REGEX", in_types=[], ret_type=TermClass)
        self.class_dict = defaultdict(lambda: defaultdict(lambda: []))

        # add primitives to primitive set. all primitives are really constructot that mutate classes
        self.pset = self.add_regex_primitives()
        self.pset = self.add_regex_terminals()


    def init_pop(self):
        pop = self.gen_first_pop_from_examples() + self.gen_first_pop_from_examples(opt=True)
        pop += pop
        if len(pop) < self.pop_size:
            ramped_random_pop = []
            for _ in range(self.pop_size - len(pop)):
                ramped_random_pop.append(self.genRampedRegexTree(self.min_ht,
                                                                 self.max_ht,
                                                                 self.term_ratio,
                                                                 self.pset,
                                                                 classes=self.classes))
            return pop + ramped_random_pop
        else:
            return random.sample(pop, self.pop_size)


    # only a very specific type of annotated data in json format with match and unmatch fields will work
    def get_data(self, filename, threshold_pct, max_ex=50):
        with open(filename) as json_file:
            data = json.load(json_file)
            char_counts = self.get_char_counts(data['examples'])
            ex_num = min(len(data['examples']), max_ex)
            data['examples'] = data['examples'][:ex_num]
            regex_target = data['regexTarget']
            num_examples = len(data['examples'])
            terminals = self.get_terminals_from_common_tokens(data['examples'], num_examples * threshold_pct)
            # print(terminals)

        return data, regex_target, terminals, num_examples, char_counts


    # constructs |examples| number of individual trees that perfectly capture their own target to seed algorithm
    def gen_first_pop_from_examples(self, opt=False):
        if opt:
            return [self.gen_best_individual(ex) for ex in self.data['examples']]
        else:
            return [self.generate_individual_from_example(ex, self.PATTERN1) for ex in self.data['examples']]

    def add_to_class_dict(self, ret_type, arg_types, constr_or_cat):
        prim = [p for p in self.pset.primitives[ret_type] if p.args == arg_types and constr_or_cat in p.name][0]
        self.class_dict[ret_type][constr_or_cat].append(prim)


    def add_regex_primitives(self):
        # term identity function for single terminal segment constructions
        classes = self.classes
        pset = self.pset

        # the OR function can take any 2 classes
        for typ1 in classes:
            for typ2 in classes:
                type_precedence = class_precedence(typ1, typ2)
                pset.addPrimitive(or_prim, [typ1, typ2], type_precedence)
                self.add_to_class_dict(type_precedence, [typ1, typ2], 'or')
                if typ1 != typ2:
                    pset.addPrimitive(or_prim, [typ2, typ1], type_precedence)
                    self.add_to_class_dict(type_precedence, [typ2, typ1], 'or')

        # the Cat function can take any 2 classes but returns an object with the highest precedence
        for typ1 in classes:
            for typ2 in classes:
                type_precedence = class_precedence(typ1, typ2)
                pset.addPrimitive(cat_prim, [typ1, typ2], type_precedence)
                self.add_to_class_dict(type_precedence, [typ1, typ2], 'cat')
                if typ1 != typ2:
                    pset.addPrimitive(cat_prim, [typ2, typ1], type_precedence)
                    self.add_to_class_dict(type_precedence, [typ2, typ1], 'cat')

        # for typ in classes:
        #   if typ != Group:
        #      pset.addPrimitive(Group, [typ], Group)


        for typ in [TermClass, ChClass, NChClass, StarClass, PlusClass]:
            pset.addPrimitive(looka_constr, [typ], LookAheadClass)
            pset.addPrimitive(lookb_constr, [typ], LookBehindClass)
            self.add_to_class_dict(LookAheadClass, [typ], 'constr')
            self.add_to_class_dict(LookBehindClass, [typ], 'constr')
            if typ != PlusClass:
                pset.addPrimitive(plus_constr, [typ], PlusClass)    # a+, a|b+, [ab]+, [^ab]+, a*+, (ab)+
                self.add_to_class_dict(PlusClass, [typ], 'constr')
                if typ != StarClass:
                    pset.addPrimitive(q_constr, [typ], QClass)
                    pset.addPrimitive(star_constr, [typ], StarClass)  # a*, a|b*, [ab]*, [^ab]*, (a)*
                    self.add_to_class_dict(StarClass, [typ], 'constr')
                    self.add_to_class_dict(QClass, [typ], 'constr')


        pset.addPrimitive(ch_constr, [TermClass], ChClass)
        pset.addPrimitive(nch_constr, [TermClass], NChClass)
        self.add_to_class_dict(ChClass, [TermClass], 'constr')
        self.add_to_class_dict(NChClass, [TermClass], 'constr')

        pset.addPrimitive(term_constr, [TermClass], TermClass)
        self.add_to_class_dict(TermClass, [TermClass], 'constr')

        return pset

    def add_regex_terminals(self):
        pset = self.pset
        # list order is used to directly reference objects by name! (invalid chat needed to be represented elsewise,
        for i, t in enumerate(self.terminals):
            pset.addTerminal(TermClass(t), TermClass, name="Term_" + str(i))


            if '[' not in t:
                pset.addTerminal(ChClass(t), ChClass, name="Ch_" + str(i))
                pset.addTerminal(NChClass(t), NChClass, name="NCh_" + str(i))
            pset.addTerminal(QClass(t), QClass, name="Q_" + str(i))
            pset.addTerminal(LookAheadClass(t), LookAheadClass, name="LookA_" + str(i))
            pset.addTerminal(LookBehindClass(t), LookBehindClass, name="LookB_" + str(i))
            if '*' not in t and '+' not in t:
                pset.addTerminal(StarClass(t), StarClass, name="Star_" + str(i))
            if '+' not in t:
                pset.addTerminal(PlusClass(t), PlusClass, name="Plus_" + str(i))
            #pset.addTerminal(OrClass(t), OrClass,
             #                name="Or_" + str(i))
            #pset.addTerminal(or_constr(TermClass(random.choice(self.terminals)), TermClass(t)), OrClass,
             #                name="Or2_" + str(i))
            # pset.addTerminal(GroupClass(t), GroupClass, name="groupcls" + str(i))


        # print(terminals)
        return pset


    def filter_tokens(self, token_counts, threshold):
        filtered_counts = {}
        for token, count in token_counts.items():
            if count >= threshold:
                filtered_counts[token] = count
        return filtered_counts

    def escape_char(self, token):
        need_escape = ['.', ',', '[', ']', '?', '{', '}', '(', ')', '|', '*', '+', '\\', '^', '@']
        return ''.join([('\\' + ch) if (ch in need_escape) else ch for ch in token])

    def get_terminals_from_common_tokens(self, examples, threshold):
        token_counts = defaultdict(int)
        for ex in examples:
            ex_str = ex['string']
            m_tokens = [v for t in [self.PATTERN1.findall(ex_str[m['start']:m['end']]) for m in ex['match']] for v in t]
            ex_tokens = [v for t in [self.PATTERN1.findall(ex_str[um['start']:um['end']]) for um in ex['unmatch']] for v in t]

            for token in set(m_tokens + ex_tokens):  # get unique token count
                token_counts[token] += 1
        token_counts = self.filter_tokens(token_counts, threshold)
        return [token for token in token_counts.keys()]

    def get_char_counts(self, examples):
        char_counts = defaultdict(int)
        for ex in examples:
            char_counts['total_matched'] += sum([(m['end'] - m['start']) for m in ex['match']])
            char_counts['total_unmatched'] += sum([(um['end'] - um['start']) for um in ex['unmatch']])
        return char_counts

    def search_terms(self, term_class, num_str):
        for t in self.pset.terminals[term_class]:
            if num_str == regex.split('[A-Za-z]+_', t.name)[1]:  # extract term from name by order of terminal list
                return t

    def get_term(self, t_class, token, w=True):
        try:
            indx = str(self.terminals.index(token))
            return self.search_terms(t_class, indx), indx
        except:
            if token.isdigit():
                if len(token) > 1:
                    return self.search_terms(t_class, '6'), '6'  # '\d+'
                else:
                    return self.search_terms(t_class, '2'), '2'  # '\d'
            elif token.isalpha():
                if len(token) > 1:
                    return (self.search_terms(t_class, '4'), '4') if w else (self.search_terms(t_class, '5'), '5')  # \w+ or [A-Za-z]+
                else:
                    return (self.search_terms(t_class, '0'), '0') if w else (self.search_terms(t_class, '1'), '1')
            else:
                return self.search_terms(t_class, '8'), '8'  # '.*'



    def collapse_expr_helper(self, expr, terms, drop_limit=10, skip_count=2):
        collapsed_expr, plus_locs = [], []
        i = 0
        while i < len(terms):
            found = False

            # TODO: find a way to finish implementing this
            # Other part of support for capturing repeat pairs, not just single terms
            if len(terms[i:]) > 5:
                if terms[i:i+2] == terms[i+2:i+4] == terms[i+4:i+6]:  # 3 pairs in a row
                    """
                    # make each term a lookahead and "or" them (since just concatenating and "plussing" didn't work)
                    looka_cat = [self.get_primitive(LookAheadClass, [TermClass])]
                    collapsed_expr += [expr[i]] + looka_cat + [expr[i+1]] + looka_cat + \
                    [self.get_primitive(OrClass, [LookAheadClass, LookAheadClass])]
                    """

                    next_pair_loc = i + 6
                    pair_count = 3
                    while terms[i:i + 2] == terms[next_pair_loc:next_pair_loc + 2]:
                        pair_count += 1
                        next_pair_loc += 2

                    if pair_count > drop_limit:
                        collapsed_expr += expr[i:i + 2]
                        i += skip_count
                        found = True

            if len(terms[i:]) > 2:
                if terms[i] == terms[i + 1] == terms[i + 2]:  # 3 symbols in a row
                    collapsed_expr += expr[i:i + 2] + [self.get_primitive(PlusClass, [TermClass])]
                    found_loc = i
                    i += 1
                    found = True

                    while terms[i] == terms[found_loc] and i < len(terms) - 1:
                        i += 1

            if not found:
                collapsed_expr.append(expr[i])
                i += 1

        return collapsed_expr

    # get the most number of drops possible before non-capture
    def get_optimal_drops(self, example, pattern, debug=False):
        non_opt_expr = self.generate_individual_from_example(example, pattern, collapse=False)
        tree_non_opt = gp.PrimitiveTree(non_opt_expr)
        ts = gp.compile(tree_non_opt, self.pset).s
        best_len, best_expr = len(ts), non_opt_expr
        target_test = regex.compile(self.regex_target).findall(example['string'])
        upper = 15
        while upper > 2:
            skip = 2
            while skip < 12:
                temp_expr = self.generate_individual_from_example(example, pattern, collapse=True, drop_limit=upper, skip_count=skip)
                tree_test = gp.PrimitiveTree(temp_expr)
                ts = gp.compile(tree_test, self.pset).s
                drop_len = len(ts)
                indr = regex.compile(ts)
                test = indr.findall(example['string'])

                if test == target_test:
                    skip += 2
                    upper -= 1
                    if drop_len < best_len:
                        if debug:
                            print(f'Regex: {indr}, length: {drop_len}, test: {test}')
                        best_expr = temp_expr
                        best_len = drop_len
                else:
                    skip = 12  # no use trying to skip more is skipping this many doesn't work
                    upper = 2  # no use trying a smaller limit if a larger doesn't work

        return best_expr


    def gen_best_individual(self, ex, w=True, collapse=False, cats=1, drop_limit=10, skip_counts=2):

        # try first match pattern which produces shorter results (combines punctuation)
        expr = self.generate_individual_from_example(ex, self.PATTERN1)
        target = regex.compile(self.regex_target).findall(ex['string'])
        expr_regex = regex.compile(gp.compile(gp.PrimitiveTree(expr), self.pset).s)
        if expr_regex.findall(ex['string']) == target:
            best_pattern = self.PATTERN1
        else:
            best_pattern = self.PATTERN2

        return self.get_optimal_drops(ex, best_pattern)


    # constructs a regex (?<= beforetarg)target(?= aftertarg) for each example
    def generate_individual_from_example(self, ex, match_pattern, w=True, collapse=False, cats=1, drop_limit=10, skip_count=2):

        split_ex_bounds = [(m.start(0), m.end(0)) for m in match_pattern.finditer(ex['string'])]

        # one cat for LB and LA, another for that and LA after T
        m_bounds = (ex['match'][0]['start'], ex['match'][0]['end'])  # process only the first match
        behind_expr, target_expr, ahead_expr = [], [], []
        b_terms, t_terms, a_terms = [], [], []
        ex_str = ex['string']

        # only handles case of single fit match! TODO: allow other, non-contiguous matches, need groups...
        for i, bounds in enumerate(split_ex_bounds):
            if i < self.max_regex_len:
                overlap = self.get_overlap(bounds, m_bounds)
                term, indx = self.get_term(TermClass, ex_str[bounds[0]:bounds[1]])
                if overlap == 0:
                    if bounds[0] >= m_bounds[1]:
                        ahead_expr.append(term)
                        a_terms.append(indx)
                    else:
                        behind_expr.append(term)
                        b_terms.append(indx)
                else:
                    target_expr.append(term)
                    t_terms.append(indx)

        if collapse:
            behind_expr = self.collapse_expr_helper(behind_expr, b_terms, drop_limit, skip_count)
            target_expr = self.collapse_expr_helper(target_expr, t_terms, drop_limit, skip_count)
            ahead_expr = self.collapse_expr_helper(ahead_expr, a_terms, drop_limit, skip_count)

        if cats == 2:
            bx = self.add_catsV2(behind_expr, collapse)
            tx = self.add_catsV2(target_expr, collapse)
            ax = self.add_catsV2(ahead_expr, collapse)
        else:
            bx = self.add_catsV1(behind_expr, collapse)
            tx = self.add_catsV1(target_expr, collapse)
            ax = self.add_catsV1(ahead_expr, collapse)


        # rather contrived way of constructing the target regex from the examples, as Bartoli, et al. also do
        ab_cap = [self.get_primitive(LookAheadClass, [LookAheadClass, LookBehindClass], key='cat')]
        bcap = [self.get_primitive(LookBehindClass, [bx[0].ret])]
        atcap = [self.get_primitive(LookAheadClass, [LookAheadClass, tx[0].ret], key='cat')]
        acap = [self.get_primitive(LookAheadClass, [ax[0].ret])]

        return ab_cap + bcap + bx + atcap + tx + acap + ax


    # arg_types must be list, or error will occur on counting characters or whatever the type is made of
    def get_primitive(self, ret_type, arg_types, key=None):
        try:
            for p in self.pset.primitives[ret_type]:
                if key:
                    if key in p.name:
                        if p.args == arg_types:
                            return p
                else:
                    if p.args == arg_types:
                        return p

        except ValueError:
            print("There are no primitives with that return type that accept those arg types in the pset.")




    # add concatenation primitives to list of (ONLY) terminals, TODO: close plus concatenations as V2...
    # creates a balanced tree
    def add_catsV1(self, expr, plus_locs):
        cat_prim = [self.get_primitive(TermClass, [TermClass, TermClass], key='cat')]
        mod_expr, depths, c, ex_len = [], [0], 0, len(expr)
        for i in range(ex_len - 1, -2, -1):
            if c % 2 == 0 and c != 0:

                mod_expr += cat_prim
                depths[0] += 1

                for j in range(len(depths)):
                    if depths[j] == 2:
                        mod_expr += cat_prim
                        depths[j] = 0
                        if j == len(depths) - 1:
                            depths += [1]
                        else:
                            depths[j + 1] += 1

            if i >= 0:
                mod_expr.append(expr[i])
            c += 1

        while len(mod_expr) < 2 * len(expr) - 1:
            mod_expr += cat_prim

        return mod_expr[::-1]



    # returns after modifying the new list of terms by 3
    def cats_collapse_closure(self, expr, mod_expr, indx, depth_stack):
        if indx > 2:

            """
            # Attempts to add in grouping of 2 repeated symbols with (?=\w|?=\s) or (\w\s)+ were unsuccessful
            # mostly due to unsupported python regex library behavior...
            if 'or' in expr[indx].name:          # a plus has occurred on a cat of two symbols as part of collapse
                if len(depth_stack) == 0:
                    mod_expr += [expr[indx - 2]] + [expr[indx - 1]] + [expr[indx - 4]] + [expr[indx - 3]]
                    depth_stack.append(expr[indx])
                else:        # depth_stack length must be 1..
                    stack_pop = depth_stack.pop()
                    stack_cat_term = self.get_primitive(OrClass, [stack_pop.ret, OrClass], cat=True)
                    mod_expr += [stack_pop] + [expr[indx - 2]] + [expr[indx - 3]] + \
                                [expr[indx - 4]] + expr[indx - 1:indx + 1]
                    depth_stack.append(stack_cat_term)
                return mod_expr, depth_stack, indx - 4
            """

            if 'plus'in expr[indx].name:
                if len(depth_stack) == 1:
                    stack_pop = depth_stack.pop()
                    stack_cat_term = self.get_primitive(class_precedence(stack_pop.ret, PlusClass),
                                                                         [stack_pop.ret, PlusClass], key='cat')
                    depth_stack.append(stack_cat_term)
                    mod_expr += [stack_pop] + [expr[indx - 1:indx + 1]]
                elif len(depth_stack) == 0:
                    mod_expr += expr[indx - 1]
                    depth_stack.append(expr[indx])
                return mod_expr, depth_stack, indx - 1


        return mod_expr, depth_stack + [expr[indx]], indx



    def add_catsV2(self, expr, collapse=False):

        mod_expr, depth_stack = [], []
        i = len(expr) - 1
        while i > -1:

            if collapse:
                mod_expr, depth_stack, i = self.cats_collapse_closure(expr, mod_expr, i, depth_stack)
            else:
                depth_stack.append(expr[i])

            if len(depth_stack) == 2:
                tL, tR = depth_stack[0], depth_stack[1]
                depth_stack = []
                priority_cat = self.get_primitive(class_precedence(tL.ret, tR.ret), [tL.ret, tR.ret], key='cat')
                depth_stack.append(priority_cat)
                mod_expr += [tL, tR]
            i -= 1

        while len(depth_stack) > 0:
            mod_expr.append(depth_stack.pop())


        return mod_expr[::-1]




    def c1(self, height, depth):
        """Expression generation stops when the depth is equal to height."""
        return depth == height

    def c2(self, height, depth, min_, ratio):
        """Expression generation stops when the depth is equal to height
        or when it is randomly determined that a node should be a terminal.
        """
        return depth == height or \
               (depth >= min_ and random.random() < ratio)

    # divide evenly between cats and contructors (don't favor the volume of cats in selection, won't be random)
    def select_primitive(self, typ):
        rnum = random.random()
        if rnum > 0.5:
            return random.choice(self.class_dict[typ]['constr'])
        elif rnum > 0.02:
            return random.choice(self.class_dict[typ]['cat'])
        else:
            return random.choice(self.class_dict[typ]['or'])

    def get_condition(self, height, depth, min_, ratio):
        return self.c1(height, depth) if random.random() > 0.5 else self.c2(height, depth, min_, ratio)




    def genRampedRegexTree(self, min_, max_, ratio, pset, type_=None, classes=None):
        """ modified from the fuctions in DEAP github gp.genHalfAndHalf, gp.genFull, gp.genenerate,
        equivalent to Ramped generation of Bartoli, et al.
        pset:      a DEAP object holding the primitives and terminals and their relationships
        min_:      a number representing the smallest height allowed in a tree
        max_:      the largest height allowed
        ratio:     the percentage of terminals the tree should roughly have (truer over larger trees)
        condition: a function parameter for which type of node to pick next
        typ_:      the top level type of the tree, the first node of the expression
        classes:   an optional list of classes to randomly choice from for the root node """

        if type_ is None:
             if classes:

                 ts = random.choice(classes)
                 type_ = [x for x in pset.terminals if str(ts) in str(x)][0]
             else:
                type_ = pset.ret

        expr = []
        height = random.randint(min_, max_)
        stack = [(0, type_)]
        while len(stack) != 0:
            depth, type_ = stack.pop()
            if self.get_condition(height, depth, min_, ratio):  # random of 2 conditions for term vs prim next
                try:
                    term = random.choice(pset.terminals[type_])

                except IndexError:
                    _, _, traceback = sys.exc_info()
                    raise IndexError("The genRampedRegexTree function tried to add " \
                                     "a terminal of type '%s', but there is " \
                                 "none available." % (type_,)).with_traceback(traceback)
                expr.append(term)
            else:
                try:

                    prim = self.select_primitive(type_)
                except IndexError:
                    _, _, traceback = sys.exc_info()
                    raise IndexError("The genRampedRegexTree function tried to add " \
                                     "a primitive of type '%s', but there is " \
                                     "none available." % (type_,)).with_traceback(traceback)
                expr.append(prim)
                for arg in reversed(prim.args):
                    stack.append((depth + 1, arg))
        return expr


    def identical_range(self, cap_str, match_strings):
    # returns 1 if the captured match segment is equal to any of target match portions, 0 otherwise
    # currently only 1 match is supported
    # match_strings is a list of target strings and cap_str is the captured string
        for m_str in match_strings:
            if cap_str == m_str:  # identical range
                return 1
        return 0


    def get_overlap(self, r1, r2):
        return max(0, min(r1[1], r2[1]) - max(r1[0], r2[0]))


    def evaluate_regex(self, individual, fit_cache=True):
        indv = gp.compile(expr=individual, pset=self.pset)
        indv = indv.s
        fit3 = len(indv)

        #print(repr(indv))
        if fit_cache:
         if indv in self.fit_cache.keys():
               fit1, fit2 = self.fit_cache[indv]
         else:
             indr = regex.compile(indv)
             tp, char_tp, fp, char_fp, tn, fn = 0, 0, 0, 0, 0, 0
             captured = []
             for ex in self.data['examples']:

                 for r in indr.findall(ex['string']):
                     if type(r) == tuple:
                         for t in r:
                             captured.append(t)
                     else:
                         captured.append(r)
                 captured = list(set(captured))

                 match_strings = [ex['string'][m['start']:m['end']] for m in ex['match']]

                 if len(captured) > 0:
                     #print(repr(ex['string']))
                     #print(captured)
                     for cap_str in captured:
                         if len(cap_str) > 0:
                             tmp_tp = self.identical_range(cap_str, match_strings)  # returns 1 or 0
                             cap_idxs = regex.finditer(regex.escape(cap_str), ex[
                                 'string'])  # get indices for all locations of FIRST captured string
                             for cap_idx in cap_idxs:
                                 if cap_idx.start() != -1:
                                     for match in ex['match']:
                                         char_tp += self.get_overlap((cap_idx.start(), cap_idx.start() + len(cap_str)),
                                                                     (match['start'], match['end']))
                                     for umatch in ex['unmatch']:
                                         tmp_char_fp = self.get_overlap(
                                             (cap_idx.start(), cap_idx.start() + len(cap_str)),
                                             (umatch['start'], umatch['end']))
                                         fp += int(tmp_char_fp > 0)
                                         char_fp += tmp_char_fp

                             fp -= tmp_tp
                             tp += tmp_tp

             fn = self.char_counts['total_matched'] - char_tp
             tn = self.char_counts['total_unmatched'] - char_fp

             fit1 = 1 - self.precision(tp, fn)
             fit2 = (self.fpr(tn, fp) + self.fnr(tp, fn)) * 100.0

             self.fit_cache[indv] = (fit1, fit2)

        return fit1, fit2, fit3

    def accuracy(self, tp, tn, fp, fn):
        return (tp + tn) / (tp + tn + fp + fn)

    def precision(self, tp, fp):
        return tp / (tp + fp)

    def recall(self, tp, fn):
        return tp / (tp + fn)

    def fpr(self, tn, fp):
        return fp / (tn + fp)

    def fnr(self, tp, fn):
        return fn / (tp + fn)

    def test_example(self, ex, optimal=False, cats=1):
        print("\nExample:")
        print(ex)

        target_test = regex.compile(self.regex_target).findall(ex['string'])
        print("Target:")
        print(target_test)


        if optimal:
            expr = self.gen_best_individual(ex)
        else:
            expr = self.generate_individual_from_example(ex, self.PATTERN1, collapse=False, cats=cats)

        if len(expr) > 0:
            tree = gp.PrimitiveTree(expr)
            ts = gp.compile(tree, self.pset).s
            indr = regex.compile(ts)
            print(indr)
            test = indr.findall(ex['string'])
            print(test)
            print(f'Expression Length: {len(ts)}')
        else:
            print("No match found.")







    def demo_test(self):
        # TESTING ...

        # Generate 10 random trees
        for i in range(10):
            pset = self.pset
            terminals = self.terminals
            expr = self.genRampedRegexTree(min_=7, max_=20, ratio=0.7, pset=self.pset, classes=self.classes)
            tree = gp.PrimitiveTree(expr)
            indr = gp.compile(tree, self.pset).s
            print(f"Tree: {tree}")
            print(f"Random Regex: {repr(indr)}")
            indr = regex.compile(indr)



        # an example of building a tree manually for a regular expression representation as a tree,
        # where the nodes alternate from bottom up- right branch, then left closer to primitive
        expr = [
                self.get_primitive(class_precedence(LookAheadClass, LookBehindClass), [LookAheadClass, LookBehindClass], key='cat'),
                self.get_primitive(LookBehindClass, [TermClass]),
                self.get_primitive(TermClass, [TermClass, TermClass], key='cat'),
                self.get_term(TermClass, "\w")[0],
                self.get_term(TermClass, "title")[0],
                self.get_primitive(class_precedence(LookAheadClass, TermClass), [LookAheadClass, TermClass], key='cat'),
                self.get_term(TermClass, "target ={stuff}")[0],
                self.get_primitive(class_precedence(LookAheadClass, TermClass), [LookAheadClass, TermClass], key='cat'),
                self.get_term(TermClass, "\w")[0],
                self.get_primitive(LookAheadClass, [TermClass]),
                self.get_primitive(TermClass, [TermClass, TermClass], key='cat'),
                self.get_term(TermClass, "\s")[0],
                self.get_term(TermClass, "year")[0]
                ]

        tree = gp.PrimitiveTree(expr)
        indr = regex.compile(gp.compile(tree, pset).s)
        print("\nManually Constructed INDR:")
        print(indr)
        eval_test = self.evaluate_regex(tree, fit_cache=None)


        ex_mod = {"string": "\ntitle a={La puerta}, 1 2 a b c\n",
                  "match": [{"start": 10, "end": 19}],
                  "unmatch": [{"start": 0, "end": 10}, {"start": 19, "end": 32}]}



        rexp = "(?<=\ntitle \w+={)\w+ \w+(?=}, \d+ \d+(?:\s)|(?:\w)\n)"
        #rexp = "(?<=\ntitle \w={)\w+ \w+(?=}, \d+ \d+( \w+)+\n)"
        rexp2 = "(?:(?<=\ntitle \w={))\w+ \w+(?=}, \d+ \d+( \w+)+\n)"

        indr = regex.compile(rexp)
        print(indr)
        test = indr.findall(ex_mod['string'])
        print(test)


        for e in [8, 35]:    # examples with punctuation at end of target requiring different match pattern
            self.test_example(self.data['examples'][e], optimal=False)
            self.test_example(self.data['examples'][e], optimal=True)

        drop_misses, non_drop_misses, length_diffs = 0, 0, 0
        for i, ex in enumerate(self.data['examples']):
            print(f"EXAMPLE {i}:")
            #print(ex['string'][ex['match'][0]['start']:ex['match'][0]['end']])
            expr1 = self.generate_individual_from_example(ex, self.PATTERN1)
            expr2 = self.gen_best_individual(ex)

            if i == random.choice(self.data['examples']):
                print(f"Random Eval, Ex {i}:")
                self.evaluate_regex(expr1)
                self.evaluate_regex(expr2)

            target = regex.compile(self.regex_target).findall(ex['string'])
            ex1_rstring = gp.compile(gp.PrimitiveTree(expr1), self.pset).s
            ex2_rstring = gp.compile(gp.PrimitiveTree(expr2), self.pset).s
            expr1_regex = regex.compile(ex1_rstring)
            expr2_regex = regex.compile(ex2_rstring)
            test1 = expr1_regex.findall(ex['string'])
            test2 = expr2_regex.findall(ex['string'])
            print(test1)
            print(test2)
            len1 = len(ex1_rstring)
            len2 = len(ex2_rstring)
            if test1 != target:
                non_drop_misses += 1
            if test2 != target:
                drop_misses += 1

            if (test1 == target) and (test2 == target):
                print(len1, len2)
                length_diffs += len1 - len2

        print(f'Non-Drop Misses: {non_drop_misses}, Drop Misses: {drop_misses}')
        print(f'Length Differences: {length_diffs}')


    def gen_pre(self):
        self.population.pop()


    def run(self, verbose=False):


        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)
        start = time.time()
        self.population = self.init_pop()
        elapsed = time.time() - start
        #print(f'elapsed seconds for population initialization: {elapsed}')

        self.population = [creator.Individual(indv) for indv in self.population]
        toolbox = base.Toolbox()


        toolbox.register("expr", self.genRampedRegexTree,
                         min_=self.min_ht,
                         max_=self.max_ht,
                         ratio=self.term_ratio,
                         classes=self.classes,
                         pset=self.pset)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=self.pset)


        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        logbook = tools.Logbook()
        logbook.header = ['gen'] + stats.fields



        for g in range(self.ngen):

            start_time = time.time()
            # get the fitnesses for every individual in the population
            fitnesses = futures.map(self.evaluate_regex, self.population)
            for indv, fits in zip(self.population, fitnesses):
                indv.fitness.values = fits

            # log and record progress
            record = stats.compile(self.population)
            logbook.record(gen=g, **record)
            if verbose:
                print(logbook.stream)
            hof.update(self.population)

            # sort by Pareto-fronts (NSGA-II, Deb, et)
            self.population = [indv for front in tools.sortNondominated(self.population, self.pop_size) for indv in front]

            keep_num = int(self.pop_size * 0.9)  # keep 90% of old gen
            new_pop = []
            while len(new_pop) < keep_num:
                rnum = random.random()
                if rnum < self.CXPB:
                    cx_indv1 = tools.selTournament(self.population, k=1, tournsize=7)[0]
                    cx_indv2 = tools.selTournament(self.population, k=1, tournsize=7)[0]

                    # cx_indv1, cx_indv2 = gp.cxOnePointLeafBiased(cx_indv1, cx_indv2, self.term_ratio)
                    cx_indv1, cx_indv2 = self.cxLeafOrSubTree(cx_indv1, cx_indv2, self.term_ratio)
                    new_pop.append(cx_indv1)
                    new_pop.append(cx_indv2)
                elif rnum < self.CXPB + self.MUTPB:
                    mutant = toolbox.mutate(tools.selTournament(self.population, k=1, tournsize=7)[0])[0]
                    new_pop.append(mutant)
                else:
                    new_pop.append(tools.selTournament(self.population, k=1, tournsize=7)[0])

            self.population = new_pop + toolbox.population(n=self.pop_size - keep_num)

            best = tools.selBest(self.population, k=1)[0]
            tree = gp.PrimitiveTree(best)
            print('Best of that gen:')
            print(gp.compile(tree, pset=self.pset).s + '\nFitness: ' + str({best.fitness.values}))
            elapsed_time = time.time() - start_time
            remaining_min = (elapsed_time * (self.ngen - g)) / 60
            remaining_hours = remaining_min / 60
            print(f"Time for last gen: {elapsed_time} secs, Remaining: {remaining_min} minutes, {remaining_hours} hours.")
            print('[' + ('*' * (g // self.ngen)) + ((100 - (g // self.ngen)) * ' ') + ']')

        return hof, logbook


    def get_prim_ret_types(self, indv):
        prim_ret_types = defaultdict(list)
        for i, e in enumerate(indv, 1):  # skip root
            if e.arity > 0:
                prim_ret_types[e.ret].append(i - 1)
        return prim_ret_types



    def cxLeafOrSubTree(self, indv1, indv2, trmpb):

        if (len(indv1) < 2 or len(indv2) < 2) or indv1 == indv2:
            return indv1, indv2

        rnum = random.random()
        if rnum < trmpb:
            # swap a random terminal with the same return type
            term_locs1 = [i - 1 for i, e in enumerate(indv1, 1) if e.arity == 0]
            term_locs2 = [i - 1 for i, e in enumerate(indv2, 1) if e.arity == 0]
            cx1_pnt = random.choice(term_locs1)
            cx2_pnt = random.choice(term_locs2)
            indv1[cx1_pnt], indv2[cx2_pnt] = indv2[cx2_pnt], indv1[cx1_pnt]
        else:
            # swap a subtree with the same root return type

            # get all the return types of all the primitives in both, and their locations
            prim_ret_types1 = self.get_prim_ret_types(indv1)
            prim_ret_types2 = self.get_prim_ret_types(indv2)
            common_types = set(prim_ret_types1.keys()) & set(prim_ret_types2.keys())
            if len(common_types) == 0:
                return indv1, indv2
            sub_root = random.choice(list(common_types))


            subroot_loc1 = random.choice(prim_ret_types1[sub_root])
            subroot_loc2 = random.choice(prim_ret_types2[sub_root])

            bounds1 = self.get_subtree(indv1, subroot_loc1)
            bounds2 = self.get_subtree(indv2, subroot_loc2)
            slice1 = slice(bounds1[0], bounds1[1])
            slice2 = slice(bounds2[0], bounds2[1])

            indv1[slice1], indv2[slice2] = indv2[slice2], indv1[slice1]

        return indv1, indv2









    def get_subtree(self, indv, root_loc):
        if len(indv) < 2:
            return indv
        arity_debt = indv[root_loc].arity
        next_loc = root_loc + 1

        while arity_debt > 0:
            next_node_or_leaf = indv[next_loc]
            arity_debt += next_node_or_leaf.arity - 1
            next_loc += 1

        return root_loc, next_loc

    def has_digit(self, s):
        return sum([ch.isdigit() for ch in s]) > 0


    def view_results(self, results):

        lab_dict = {'cat_prim':'+', 'q_constr':'?', 'plus_constr':'+', 'or_prim':'|', 'star_constr':'x',
                    'looka_constr':'(?=)','lookb_constr':'(?<=)', 'nch_constr':'[^ ]', 'ch_constr':'[ ]',
                    'term_constr':'t()'}
        hof, log = results
        avgs = [l['avg'] for l in log]
        plt.plot(avgs)
        plt.xlabel('Generations')
        plt.ylabel('Fitness Averages')
        plt.title('Capture Fitness Avgs over Generations (0 is perfect)')
        plt.show()

        expr = hof.__dict__['items'][0]
        tree = gp.PrimitiveTree(expr);
        str(tree) + '   =   ' + gp.compile(tree, pset=self.pset).s

        # for i in range(3):
        nodes, edges, labels = gp.graph(expr)
        print(labels.values())
        new_labs = {}
        for i, lab in enumerate(labels.values()):
            if self.has_digit(lab):
                new_labs[i] = self.pset.context[lab].s
                if new_labs[i] == ' ':
                    new_labs[i] = '\' \''
            else:
                new_labs[i] = lab_dict[lab]

        print(labels)
        print(new_labs)
        graph = networkx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        pos = graphviz_layout(graph, prog="dot")

        plt.figure(figsize=(10, 10))
        networkx.draw_networkx_nodes(graph, pos, node_size=400, node_color='w')
        networkx.draw_networkx_edges(graph, pos, edge_color="blue")
        networkx.draw_networkx_labels(graph, pos, new_labs)
        plt.axis("off")
        plt.show()



if __name__ == '__main__':
    filename = '/Volumes/My_Passport/clusters/proj/bartoli_data/Bibtex_Title.json'


    RG = RegexGenerator(filename, threshold=0.8, max_ex=100, max_regex_len=80, ngen=500, pop_size=1000)

    #RG.demo_test()
    results = RG.run(verbose=True)
    RG.view_results(results)






