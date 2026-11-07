//! Serialization fixes for profiles synthesized by moxcms, shared with the
//! bundle generator. Tag payloads retain their exact colorimetric bytes.
extern crate alloc;
use self::alloc::{collections::BTreeMap, vec::Vec};

pub(crate) fn canonicalize(icc: Vec<u8>) -> Option<Vec<u8>> {
    let read = |offset: usize| -> Option<usize> {
        Some(u32::from_be_bytes(icc.get(offset..offset.checked_add(4)?)?.try_into().ok()?) as usize)
    };
    if icc.get(36..40)? != b"acsp" || read(0)? != icc.len() {
        return None;
    }
    let count = read(128)?;
    let table_end = 132_usize.checked_add(count.checked_mul(12)?)?;
    let mut output = icc.get(..table_end)?.to_vec();
    // ICC.1:2022 7.2.16: the PCS illuminant is D50, even when media white
    // differs. Colorant/adaptation tags already describe the D50 PCS.
    output[68..80].copy_from_slice(&[
        0x00, 0x00, 0xf6, 0xd6, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0xd3, 0x2d,
    ]);
    output[24..36].fill(0); // Reproducible creation timestamp.
    output[84..100].fill(0); // No stale profile ID after changing serialization.
    let mut offsets = BTreeMap::new();
    for index in 0..count {
        let entry = 132 + index * 12;
        let offset = read(entry + 4)?;
        let size = read(entry + 8)?;
        if offset < table_end {
            return None;
        }
        let payload = icc.get(offset..offset.checked_add(size)?)?;
        let next = if let Some(&shared) = offsets.get(&(offset, size)) {
            shared
        } else {
            output.resize(output.len().checked_add(3)? & !3, 0);
            let next = u32::try_from(output.len()).ok()?;
            output.extend_from_slice(payload);
            offsets.insert((offset, size), next);
            next
        };
        output[entry + 4..entry + 8].copy_from_slice(&next.to_be_bytes());
    }
    output.resize(output.len().checked_add(3)? & !3, 0);
    let length = u32::try_from(output.len()).ok()?;
    output[..4].copy_from_slice(&length.to_be_bytes());
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repacking_preserves_shared_payloads_and_aligns_every_tag() {
        let mut source = alloc::vec![0_u8; 185];
        source[..4].copy_from_slice(&185_u32.to_be_bytes());
        source[36..40].copy_from_slice(b"acsp");
        source[128..132].copy_from_slice(&3_u32.to_be_bytes());
        for (index, (signature, offset, size)) in [
            (*b"test", 169_u32, 5_u32),
            (*b"copy", 169, 5),
            (*b"cprt", 174, 7),
        ]
        .into_iter()
        .enumerate()
        {
            let entry = 132 + index * 12;
            source[entry..entry + 4].copy_from_slice(&signature);
            source[entry + 4..entry + 8].copy_from_slice(&offset.to_be_bytes());
            source[entry + 8..entry + 12].copy_from_slice(&size.to_be_bytes());
        }
        source[169..174].copy_from_slice(b"abcde");
        source[174..181].copy_from_slice(b"payload");
        let result = canonicalize(source).unwrap();
        let read =
            |offset| u32::from_be_bytes(result[offset..offset + 4].try_into().unwrap()) as usize;
        assert_eq!(read(0), result.len());
        assert!(result.len().is_multiple_of(4));
        assert_eq!(read(136), read(148));
        for entry in [132, 144, 156] {
            assert!(read(entry + 4).is_multiple_of(4));
        }
        assert_eq!(&result[read(136)..read(136) + 5], b"abcde");
        assert_eq!(&result[read(160)..read(160) + 7], b"payload");
        assert_eq!(
            &result[68..80],
            &[0, 0, 246, 214, 0, 1, 0, 0, 0, 0, 211, 45]
        );
        assert_eq!(canonicalize(result.clone()).unwrap(), result);
    }

    #[test]
    fn malformed_tag_bounds_are_refused() {
        assert!(canonicalize(alloc::vec![0; 132]).is_none());
        let mut bytes = alloc::vec![0; 144];
        bytes[..4].copy_from_slice(&144_u32.to_be_bytes());
        bytes[36..40].copy_from_slice(b"acsp");
        bytes[128..132].copy_from_slice(&1_u32.to_be_bytes());
        bytes[136..140].copy_from_slice(&u32::MAX.to_be_bytes());
        bytes[140..144].copy_from_slice(&8_u32.to_be_bytes());
        assert!(canonicalize(bytes).is_none());
    }
}
