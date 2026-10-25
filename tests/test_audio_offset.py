import numpy as np

from ai_furby.audio import remove_offset


def test_remove_offset_centres_the_block_on_zero():
    block = np.array([0.5, 0.7, 0.6, 0.4], dtype=np.float32)
    cleaned = remove_offset(block)
    assert abs(float(cleaned.mean())) < 1e-6


def test_remove_offset_keeps_the_variation():
    block = np.array([1.0, 3.0, 1.0, 3.0], dtype=np.float32)
    cleaned = remove_offset(block)
    assert np.allclose(cleaned, [-1.0, 1.0, -1.0, 1.0])


def test_remove_offset_handles_empty_blocks():
    assert remove_offset(np.zeros(0, dtype=np.float32)).size == 0
