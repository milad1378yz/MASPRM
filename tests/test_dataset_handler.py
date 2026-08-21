from dataset_handler import _remove_soft_newlines


def test_remove_soft_newlines_preserves_regular_newlines():
    soft_newlines = "".join(map(chr, (11, 12, 13, 0x85, 0x2028, 0x2029)))

    assert _remove_soft_newlines(f"before\n{soft_newlines}after") == "before\nafter"
