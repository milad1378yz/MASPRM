from dataset_handler import _remove_soft_newlines


def test_remove_soft_newlines_preserves_regular_newlines():
    soft_newlines = "".join(chr(codepoint) for codepoint in (0x000B, 0x000C, 0x000D, 0x0085, 0x2028, 0x2029))

    assert _remove_soft_newlines(f"before\n{soft_newlines}after") == "before\nafter"
