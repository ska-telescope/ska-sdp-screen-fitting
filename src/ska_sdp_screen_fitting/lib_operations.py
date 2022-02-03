"""
    Some utilities for operations
    SPDX-License-Identifier: BSD-3-Clause
"""

import multiprocessing

import numpy as np

from ska_sdp_screen_fitting._logging import logger as logging


class MultiprocManager:
    """
    This class is a manager for multiprocessing
    """

    class MultiThread(multiprocessing.Process):
        """
        This class is a working thread which load parameters from a queue and
        return in the output queue
        """

        def __init__(self, in_queue, out_queue, funct):
            multiprocessing.Process.__init__(self)
            self.in_queue = in_queue
            self.out_queue = out_queue
            self.funct = funct

        def run(self):

            while True:
                parms = self.in_queue.get()

                # poison pill
                if parms is None:
                    self.in_queue.task_done()
                    break

                self.funct(*parms, out_queue=self.out_queue)
                self.in_queue.task_done()

    def __init__(self, procs=0, funct=None):
        """
        Manager for multiprocessing
        procs: number of processors, if 0 use all available
        funct: function to parallelize / note that the last parameter of this
        function must be the out_queue
        and it will be linked to the output queue
        """
        if procs == 0:
            procs = multiprocessing.cpu_count()
        self.procs = procs
        self._threads = []
        self.in_queue = multiprocessing.JoinableQueue()
        self.out_queue = multiprocessing.Queue()
        self.runs = 0

        logging.debug("Spawning %i threads...", self.procs)
        for _ in range(self.procs):
            thread = self.MultiThread(self.in_queue, self.out_queue, funct)
            self._threads.append(thread)
            thread.start()

    def put(self, args):
        """
        Parameters to give to the next jobs sent into queue
        """
        self.in_queue.put(args)
        self.runs += 1

    def get(self):
        """
        Return all the results as an iterator
        """
        # NOTE: do not use queue.empty() check which is unreliable
        # https://docs.python.org/2/library/multiprocessing.html
        for _ in range(self.runs):
            yield self.out_queue.get()

    def wait(self):
        """
        Send poison pills to jobs and wait for them to finish
        The join() should kill all the processes
        """
        for _ in self._threads:
            self.in_queue.put(None)

        # wait for all jobs to finish
        self.in_queue.join()
        self.in_queue.close()

    def __del__(self):
        for thread in self._threads:
            thread.terminate()
            del thread


def reorder_axes(array, old_axes, new_axes):
    """
    Reorder axis of an array to match a new name pattern.

    Parameters
    ----------
    array : np array
        The array to transpose.
    old_axes : list of str
        A list like ['time','freq','pol'].
        It can contain more axes than the new list, those are ignored.
        This is to pass to oldAxis the soltab.getAxesNames() directly even on
        an array from getValuesIter()
    new_axes : list of str
        A list like ['time','pol','freq'].

    Returns
    -------
    np array
        With axis transposed to match the new_axes list.
    """
    old_axes = [ax for ax in old_axes if ax in new_axes]
    idx = [old_axes.index(ax) for ax in new_axes]
    return np.transpose(array, idx)


def remove_keys(dic, keys=[]):
    """
    Remove a list of keys from a dict and return a new one.

    Parameters
    ----------
    dic : dcit
        The input dictionary.
    keys : list of str
        A list of arguments to remove or a string for single argument.

    Returns
    -------
    dict
        Dictionary with removed keys.
    """
    dic_copy = dict(dic)
    if isinstance(keys, str):
        keys = [keys]
    for key in keys:
        del dic_copy[key]
    return dic_copy


def normalize_phase(phase):
    """
    Normalize phase to the range [-pi, pi].

    Parameters
    ----------
    phase : array of float
        Phase to normalize.

    Returns
    -------
    array of float
        Normalized phases.
    """

    # Convert to range [-2*pi, 2*pi].
    out = np.fmod(phase, 2.0 * np.pi)
    # Remove nans
    nans = np.isnan(out)
    np.putmask(out, nans, 0)
    # Convert to range [-pi, pi]
    out[out < -np.pi] += 2.0 * np.pi
    out[out > np.pi] -= 2.0 * np.pi
    # Put nans back
    np.putmask(out, nans, np.nan)
    return out
