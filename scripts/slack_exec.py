#!/usr/bin/env python3
"""Set this process's timer slack to 1 ns, then exec the command: threads inherit it."""
import ctypes
import os
import sys
ctypes.CDLL(None).prctl(29, 1, 0, 0, 0)  # PR_SET_TIMERSLACK
os.execvp(sys.argv[1], sys.argv[1:])
