import logging
import unittest

from piv2hdf import set_loglevel, logger


class TestLogger(unittest.TestCase):

    def test_logger(self):

        for loglevel in (logging.CRITICAL, logging.ERROR, logging.WARN, logging.ERROR, logging.INFO):
            set_loglevel(loglevel)
            self.assertEqual(logger.level, loglevel)
            for handler in logger.handlers:
                self.assertEqual(handler.level, loglevel)
