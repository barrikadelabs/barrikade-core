from core.layer_a.unicode import normalize_uniccode

def test_normalize_unicode():
    """Test the normalize_uniccode function."""
    samples = [
        ("Café", "Café"),  # Already normalized
        ("Cafe\u0301", "Café"),  # 'e' + combining acute accent
        ("\u212B", "Å"),  # Angstrom sign to A with ring
        ("ﬁ", "fi"),  # Ligature to separate letters
    ]
    
    for original, expected in samples:
        normalized = normalize_uniccode(original)
        assert normalized == expected, f"Expected {expected}, got {normalized}"
        print(f"Original: {original} | Normalized: {normalized}")

if __name__ == "__main__":
    test_normalize_unicode()
