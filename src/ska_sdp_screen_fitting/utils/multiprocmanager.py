"""
    This script contains the MultiprocManager, to handle multiprocessing
    SPDX-License-Identifier: BSD-3-Clause
"""

import multiprocessing

from ska_sdp_screen_fitting.utils._logging import logger as logging


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
