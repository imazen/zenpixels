use super::murmur3::{Murmur3_32, murmur3_32};

// Produced by the independent SMHasher C++ implementation, never by the Rust code.
const GOLDEN: &str = include_str!("murmur3.txt");

#[test]
fn c_golden_pairs_and_every_split() {
    for line in GOLDEN.lines().filter(|s| !s.starts_with('#')) {
        let parts: Vec<_> = line.split_whitespace().collect();
        assert_eq!(parts.len(), 3);
        let seed = u32::from_str_radix(parts[0], 16).unwrap();
        let len: usize = parts[1].parse().unwrap();
        let expected = u32::from_str_radix(parts[2], 16).unwrap();
        let data: Vec<u8> = (0..len).map(|i| (i * 37 + 11) as u8).collect();
        assert_eq!(murmur3_32(&data, seed), expected, "len={len} seed={seed}");
        #[cfg(feature = "hash")]
        assert_eq!(zenhash::murmur3_32(&data, seed), expected);
        // All short splits, every tail position at both ends of long inputs.
        for split in (0..=len.min(68)).chain(len.saturating_sub(68)..=len) {
            let mut h = Murmur3_32::with_seed(seed);
            h.update(&data[..split]);
            let before = h.digest();
            assert_eq!(before, h.digest(), "digest must not mutate state");
            h.update(&[]);
            h.update(&data[split..]);
            assert_eq!(h.digest(), expected, "len={len} split={split} seed={seed}");
        }
        let mut h = Murmur3_32::with_seed(seed);
        for byte in &data {
            h.update(core::slice::from_ref(byte));
        }
        assert_eq!(h.digest(), expected, "byte updates len={len}");
        // Explicitly unaligned starts exercise the same byte order.
        let mut padded = vec![0u8];
        padded.extend_from_slice(&data);
        assert_eq!(murmur3_32(&padded[1..], seed), expected);
    }
}

#[test]
fn quickwit_published_pairs_and_const_evaluation() {
    const HASH: u32 = murmur3_32(b"abc", 3_242_157_231);
    assert_eq!(HASH, 476_091_040);
    for (input, expected) in [
        (&b""[..], 36_859_204),
        (b"a", 3_144_985_375),
        (b"ab", 3_262_304_301),
        (b"abc", 476_091_040),
        (b"abcd", 412_992_581),
        (b"abcde", 2_747_833_956),
        (b"abcdefghijklmnop", 2_078_305_053),
    ] {
        assert_eq!(murmur3_32(input, 3_242_157_231), expected);
    }
}
