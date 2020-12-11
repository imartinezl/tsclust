import numpy as np
import tsclust.window


def test_no_window():
    window = tsclust.window.no_window
    query_size, reference_size, window_size = 200, 300, None
    mask = tsclust.window.precompute(window, query_size, reference_size, window_size)
    np.testing.assert_equal(mask, np.ones((query_size, reference_size)))


def test_precompute():
    window = tsclust.window.sakoe_chiba_window
    query_size, reference_size, window_size = 200, 300, 50
    mask = tsclust.window.precompute(window, query_size, reference_size, window_size)
    # tsclust.window.display(mask)
