use zenpixels::ChannelType;
use zenpixels::sample::{SampleEncoding, SampleEncodingError};

#[test]
fn accepts_exactly_the_unsigned_word_placements_that_fit() {
    for (storage, storage_bits) in [(ChannelType::U8, 8_u16), (ChannelType::U16, 16)] {
        for bits in 0..=u8::MAX {
            for shift in 0..=u8::MAX {
                let result = SampleEncoding::new(storage, bits, shift);
                let fits = bits > 0 && u16::from(bits) + u16::from(shift) <= storage_bits;
                assert_eq!(result.is_ok(), fits, "{storage:?}/{bits}/{shift}");
                if let Ok(encoding) = result {
                    assert_eq!(encoding.storage(), storage);
                    assert_eq!(encoding.code_bits(), bits);
                    assert_eq!(encoding.bit_shift(), shift);
                }
            }
        }
    }
    for storage in [ChannelType::F16, ChannelType::F32] {
        assert_eq!(
            SampleEncoding::new(storage, 10, 0),
            Err(SampleEncodingError::UnsupportedStorage)
        );
    }
}

#[test]
fn native_packed_and_normalized_words_have_distinct_interpretations() {
    for bits in [8, 10, 12] {
        let native = SampleEncoding::new(ChannelType::U16, bits, 0).unwrap();
        let packed = SampleEncoding::new(ChannelType::U16, bits, 16 - bits).unwrap();
        assert_ne!(native, packed);
        let full_word = SampleEncoding::new(ChannelType::U16, 16, 0).unwrap();
        assert_ne!(native, full_word);
        assert_ne!(packed, full_word);

        let mask = (1_u32 << bits) - 1;
        for code in 0..=mask {
            let word = code << packed.bit_shift();
            assert_eq!((word >> packed.bit_shift()) & mask, code);
            let padding = 0xffff ^ (mask << packed.bit_shift());
            assert_eq!(((word | padding) >> packed.bit_shift()) & mask, code);
        }
    }
}

#[test]
fn integer_encoding_construction_is_const() {
    const P010: SampleEncoding = match SampleEncoding::new(ChannelType::U16, 10, 6) {
        Ok(value) => value,
        Err(_) => panic!("valid P010 component"),
    };
    assert_eq!(P010.code_bits(), 10);
    assert_eq!(P010.bit_shift(), 6);
}
