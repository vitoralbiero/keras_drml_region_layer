import numpy as np
import unittest
import os


if __name__ == '__main__':
    # disable tensorflow debugging logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    np.random.seed(42)
    test_suite = unittest.TestLoader().discover(start_dir='tests',
                                                pattern='*_test.py')
    unittest.TextTestRunner(verbosity=1).run(test_suite)
