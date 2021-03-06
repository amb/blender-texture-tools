# l.py - A tiny interpreter for a lisp-like language with full
# lexical closures in Python
#
# Written by Ron Garret.  Contributed to the public domain
# modification 2021 ambi, also public domain

# The global environment
globalenv = {}


# Find the lexical frame where a variable is bound
def findframe(s, env):
    if isinstance(env, dict):
        return env
    else:
        if env[0].get(s) != None:
            return env[0]
        else:
            return findframe(s, env[1])


# Set the value of a variable.
def set(s, val, env):
    findframe(s, env)[s] = val


# The interpreter proper
class closure:
    def __init__(self, env, name, args, body):
        self.env = env
        self.name = name
        self.args = args
        self.body = body

    def __repr__(self):
        return "<closure %s (%s) %s %s>" % (self.name, self.args, self.body, self.env)

    def apply(self, params):
        frame = dict(zip(self.args, params))
        frame[self.name] = self
        return ev(self.body, [frame, self.env])


class primop:
    def __init__(self, op, name=None):
        if name == None:
            name = op.__name__
        if not callable(op):
            raise "%s is not callable" % op
        self.op = op
        globalenv[name] = self

    def __repr__(self):
        return "<primop %s>" % self.op.__name__

    def apply(self, params):
        assert len(params) == 2
        return self.op(*params)


def ev(l, env):
    if type(l) == str:
        return findframe(l, env)[l]
    elif type(l) == list:
        if len(l) == 0:
            return l
        elif l[0] == "fn":
            return closure(env, l[1], l[2], l[3])
        elif l[0] == "quote":
            return l[1:]
        elif l[0] == "cond":
            for clause in l[1:]:
                if ev(clause[0], env):
                    return ev(clause[1], env)
            return 0
        elif len(l) == 3 and l[1] == "=":
            set(l[0], ev(l[2], env), env)
        else:
            l = list(map(lambda x, env=env: ev(x, env), l))
            try:
                f = l[0].apply
            except:
                raise "%s is not a function object" % l[0]
            return f(l[1:])
    else:
        return l


# That's it!  Now we need a parser because Python doesn't provide one
import re


def parse(s):
    return lparse(re.split("(\[|\])|[\\s+|,]", s))


def parseAtom(s):
    try:
        s = int(s)
    except ValueError:
        try:
            s = float(s)
        except ValueError:
            pass
    return s


def lparse(l):
    result = []
    rstack = [result]
    for item in l:
        if item == None or item == "":
            continue
        elif item == "[":
            r1 = []
            rstack[0].append(r1)
            rstack = [r1] + rstack
        elif item == "]":
            if rstack == []:
                print("Ignoring extra right paren")
            rstack = rstack[1:]
        else:
            rstack[0].append(parseAtom(item))
    if len(rstack) > 1:
        print("Providing %s missing right parens" % (len(rstack) - 1))
    return result


def evl(s):
    return ev(parse(s), globalenv)


# Examples.  Note: FN is like LAMBDA except that it takes the name
# of the function as its first argument so it can be printed to help
# in debugging.

import operator as ops

for op in [ops.add, ops.mul, ops.sub, ops.abs, ops.mod]:
    primop(op)

primop(ops.truediv, name="div")


def eql(x, y):
    return x == y


primop(eql, "==")

if __name__ == "__main__":
    evl("[fn foo [x y] [add x y]] 2 3.3")
    evl("fact = [fn fact [x] [cond [[== x 0] 1] [1 [mul x [fact [sub x 1]]]]]]")
    print(evl("fact 3"))
    evl("pz = [fn pz [] 5]")
    print(evl("pz"))

    evl(
        """
    fib = [fn fib [x] [cond [[== x 0] 1]
                            [[== x 1] 1]
                            [1 [add [fib [sub x 1]]
                                    [fib [sub x 2]]]]]]
                                    """
    )

    print(globalenv)
