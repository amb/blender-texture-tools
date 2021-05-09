"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import bpy
import cProfile
import pstats
import io
import tracemalloc


# TODO:
# implement proper handling of "from X import Y as Z" handling
# instead of just a few specific cases
def keep_updated(lc, libs, verbose=False):
    nm = lc["__name__"]
    ilib = lc["importlib"]
    for l in libs:
        # parse path
        o = l.split("/")
        m = None
        p = ""
        if len(o) == 2:
            p, m = o
        else:
            m = o[0]
        assert m is not None

        # parse alias
        o = m.split("@")
        load_as = o[0]
        if len(o) == 2:
            m, load_as = o
        else:
            m = o[0]

        # import
        if load_as not in lc:
            if verbose:
                print("[bpy_amb.keep_updated] Import:", ("." + m, nm + p))
            temp = ilib.import_module("." + m, nm + p)
            lc[load_as] = temp
        else:
            if verbose:
                print("[bpy_amb.keep_updated] Reload:", load_as)
            lc[load_as] = ilib.reload(lc[load_as])


def install_lib(libname):
    from subprocess import call

    pp = bpy.app.binary_path_python
    call([pp, "-m", "ensurepip", "--user"])
    call([pp, "-m", "pip", "install", "--user", libname])


def profiling_start():
    # profiling
    pr = cProfile.Profile()
    pr.enable()
    return pr


def profiling_end(pr, lines=20, sortby="cumulative"):
    # end profile, print results
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.strip_dirs().sort_stats(sortby).print_stats(lines)
    print(s.getvalue())


# with Profile_this
class Profile_this:
    def __init__(self, lines=20, sortby="cumulative"):
        self.profile = profiling_start()
        self.lines = lines
        self.sortby = sortby

    def __enter__(self):
        return self.profile

    def __exit__(self, type, value, traceback):
        profiling_end(self.profile, self.lines, self.sortby)


# with Mode_set
class Mode_set:
    def __init__(self, mode):
        self.prev_mode = bpy.context.object.mode
        self.changed = True

        if self.prev_mode != mode:
            bpy.ops.object.mode_set(mode=mode)
        else:
            self.changed = False

    def __enter__(self):
        return self.prev_mode

    def __exit__(self, type, value, traceback):
        if self.changed:
            bpy.ops.object.mode_set(mode=self.prev_mode)


tracemalloc_start = None


def memorytrace_start():
    global tracemalloc_start
    if tracemalloc_start == None:
        print("Begin memorytrace")
        tracemalloc.start()
        tracemalloc_start = tracemalloc.take_snapshot()


def memorytrace_print():
    global tracemalloc_start
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.compare_to(tracemalloc_start, "lineno")
    for stat in top_stats[:10]:
        print(stat)
