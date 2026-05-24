from core.layer_a.normalise_punctuation import collapse_separated_characters

def test_collapse_separated_characters():
    """Test the collapse_separated_characters function."""
    samples = [
        ("I g n o r e", "Ignore"),
        ("I-g-n-o-r-e", "Ignore"),
        ("I.g.n.o.r.e", "Ignore"),
    ]
    
    for original, expected in samples:
        collapsed = collapse_separated_characters(original)
        assert collapsed == expected, f"Expected {expected}, got {collapsed}"
        print(f"Original: {original} | Collapsed: {collapsed}")

if __name__ == "__main__":
    test_collapse_separated_characters()
