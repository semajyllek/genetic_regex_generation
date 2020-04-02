
# classes for strictly typing what parameter the functions creating primitives can take,
# even though they are all the same, they can be recognized as different types by their __class__
class OrClass:
    def __init__(self, t):
        self.s = t


class TermClass:
    def __init__(self, t):
        self.s = t


class ChClass:
    def __init__(self, t):
        self.s = t


class PlusClass:
    def __init__(self, t):
        self.s = t


class NChClass:
    def __init__(self, t):
        self.s = t


class QClass:
    def __init__(self, t):
        self.s = t


class StarClass:
    def __init__(self, t):
        self.s = t


class LookAheadClass:
    def __init__(self, t):
        self.s = t


class LookBehindClass:
    def __init__(self, t):
        self.s = t


class GroupClass:
    def __init__(self, t):
        self.s = t



def class_precedence(typ1, typ2):
    if (typ1 == QClass) or (typ2 == QClass):
        return QClass
    elif (typ1 == LookAheadClass) or (typ2 == LookAheadClass):
        return LookAheadClass
    elif (typ1 == LookBehindClass) or (typ2 == LookBehindClass):
        return LookBehindClass
    elif (typ1 == StarClass) or (typ2 == StarClass):
        return StarClass
    elif (typ1 == PlusClass) or (typ2 == PlusClass):
        return PlusClass

    elif (typ1 == ChClass) or (typ2 == ChClass):
        return ChClass
    elif (typ1 == NChClass) or (typ2 == NChClass):
        return NChClass
    elif (typ1 == OrClass) or (typ2 == OrClass):
        return OrClass
    else:
        return TermClass


def has_star_or_plus(t):
    return True if (t.s[-1] == '*' or t.s[-1] == '+') else False



# primitive functions for building individual trees
def term_constr(term_cls):
    return term_cls


def cat_prim(t1, t2):
    return class_precedence(t1, t2)(t1.s + t2.s)


def or_prim(t1, t2):
    return class_precedence(t1, t2)(t1.s + '|' + t2.s)


def star_constr(t):
    return StarClass('(' + t.s + ')*') if has_star_or_plus(t) else StarClass(t.s + '*')


def plus_constr(t):
    return PlusClass('(' + t.s + ')+') if has_star_or_plus(t) else PlusClass(t.s + '+')


def ch_constr(t):
    return ChClass('[' + t.s + ']')


def nch_constr(t):
    return NChClass('[^' + t.s + ']')


def q_constr(t):
    return QClass('(' + t.s + ')?') if has_star_or_plus(t) else QClass(t.s + '?')


def lookb_constr(t):
    return LookBehindClass('(?<=' + t.s + ')')


def looka_constr(t):
    return LookAheadClass('(?=' + t.s + ')')


def group_constr(t):
    return GroupClass('(' + t.s + ')')

