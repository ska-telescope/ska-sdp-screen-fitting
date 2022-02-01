"""
    Logging module.
    This module generates logs
"""


import logging
import os
import sys
import time


class _ColorStreamHandler(logging.StreamHandler):

    DEFAULT = "\x1b[0m"
    RED = "\x1b[31m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    PURP = "\x1b[35m"

    CRITICAL = RED
    ERROR = RED
    WARNING = YELLOW
    INFO = GREEN
    DEBUG = PURP

    @classmethod
    def _get_color(cls, level):
        if level >= logging.CRITICAL:
            return cls.CRITICAL
        if level >= logging.ERROR:
            return cls.ERROR
        if level >= logging.WARNING:
            return cls.WARNING
        if level >= logging.INFO:
            return cls.INFO
        if level >= logging.DEBUG:
            return cls.DEBUG
        return cls.DEFAULT

    def __init__(self, stream=None):
        logging.StreamHandler.__init__(self, stream)

    def format(self, record):
        color = self._get_color(record.levelno)
        record.msg = color + record.msg + self.DEFAULT
        return logging.StreamHandler.format(self, record)


class Logger:
    """
    Logger class
    """

    def __init__(self, level="info", logfile=None, log_dir=None):

        self.logfile = logfile
        self.log_dir = log_dir
        self.backup(logfile, log_dir)
        self.set_logger(logfile)
        self.set_level(level)

    def backup(self, logfile, log_dir):
        """
        Logger backup function
        """
        if self.logfile is not None:
            # bkp old log dir
            if os.path.isdir(log_dir):
                current_time = time.localtime()
                log_dir_old = time.strftime(
                    log_dir + "_bkp_%Y-%m-%d_%H:%M", current_time
                )
                os.system(f"mv {log_dir} {log_dir_old}")
            os.makedirs(log_dir)

            # bkp old log file
            if os.path.exists(logfile):
                current_time = time.localtime()
                logfile_old = time.strftime(
                    logfile + "_bkp_%Y-%m-%d_%H:%M", current_time
                )
                os.system(f"mv {logfile} {logfile_old}")

    def set_logger(self, logfile):
        """
        Set logger
        """
        self.logger = logging.getLogger("LoSoTo")
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
        )

        # create file handler which logs even debug messages
        if self.logfile is not None:
            handler_file = logging.FileHandler(logfile)
            # handler_file.setLevel(logging.DEBUG)
            handler_file.setFormatter(formatter)
            self.logger.addHandler(handler_file)

        # create console handler with a higher log level
        handler_console = _ColorStreamHandler(stream=sys.stdout)
        # handler_console.setLevel(logging.INFO)
        handler_console.setFormatter(formatter)
        self.logger.addHandler(handler_console)

    def set_level(self, level):
        """
        Set logger level
        """

        if level == "warning":
            self.logger.setLevel(logging.WARNING)
        elif level == "info":
            self.logger.setLevel(logging.INFO)
        elif level == "debug":
            self.logger.setLevel(logging.DEBUG)
        else:
            print(f"Debug level {level} doesn't exist.")


# this is used by all libraries for logging
logger = logging.getLogger("LoSoTo")
