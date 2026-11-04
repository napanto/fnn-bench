"""Small process helper: a timed-out profiler must not leave its GPU child alive."""
import os
import signal
import subprocess


def run_command(command, output, timeout, env=None):
    proc=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGTERM)
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL)
            proc.wait()
        output.write('\nTIMEOUT: process group terminated\n')
        output.flush()
        return 124
