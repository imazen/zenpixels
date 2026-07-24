use zenpixels_convert::adapt::{
    adapt_for_encode_cow, adapt_for_encode_explicit_cow, adapt_for_encode_with_intent_cow,
};
use zenpixels_convert::{ConvertIntent, ConvertOptions, PixelCow, PixelDescriptor};

#[test]
fn packed_exact_match_borrows_original_storage() {
    let data = [1, 2, 3, 4, 5, 6];
    let adapted = adapt_for_encode_cow(
        &data,
        PixelDescriptor::RGB8_SRGB,
        2,
        1,
        6,
        &[PixelDescriptor::RGB8_SRGB],
    )
    .unwrap();

    assert!(adapted.is_borrowed());
    let slice = adapted.as_slice();
    assert_eq!(slice.as_strided_bytes().as_ptr(), data.as_ptr());
    assert_eq!((slice.width(), slice.rows(), slice.stride()), (2, 1, 6));
}

#[test]
fn strided_exact_match_borrows_without_repacking() {
    let data = [
        1, 2, 3, 4, 5, 6, 90, 91, 92, //
        7, 8, 9, 10, 11, 12, 93, 94, 95,
    ];
    let adapted = adapt_for_encode_with_intent_cow(
        &data,
        PixelDescriptor::RGB8_SRGB,
        2,
        2,
        9,
        &[PixelDescriptor::RGB8_SRGB],
        ConvertIntent::Fastest,
    )
    .unwrap();

    assert!(adapted.is_borrowed());
    let slice = adapted.as_slice();
    assert_eq!(slice.as_strided_bytes().as_ptr(), data.as_ptr());
    assert_eq!(slice.stride(), 9);
    assert_eq!(slice.row(1), &[7, 8, 9, 10, 11, 12]);
}

#[test]
fn conversion_returns_one_owned_pixel_buffer() {
    let data = [10, 20, 30, 40, 50, 60];
    let adapted = adapt_for_encode_cow(
        &data,
        PixelDescriptor::RGB8_SRGB,
        2,
        1,
        6,
        &[PixelDescriptor::RGBA8_SRGB],
    )
    .unwrap();

    assert!(adapted.is_owned());
    let slice = adapted.as_slice();
    assert_eq!(slice.descriptor(), PixelDescriptor::RGBA8_SRGB);
    assert_eq!(slice.row(0), &[10, 20, 30, 255, 40, 50, 60, 255]);
}

#[test]
fn misaligned_wide_exact_match_is_a_typed_error_not_a_panic() {
    let storage = [0u8; 7];
    let misaligned = &storage[1..];
    let result = adapt_for_encode_cow(
        misaligned,
        PixelDescriptor::RGB16,
        1,
        1,
        6,
        &[PixelDescriptor::RGB16],
    );

    assert!(result.is_err());
}

#[test]
fn explicit_variant_uses_the_same_borrowing_contract() {
    let data = [1, 2, 3, 4, 5, 6, 0, 0, 0];
    let adapted = adapt_for_encode_explicit_cow(
        &data,
        PixelDescriptor::RGB8_SRGB,
        2,
        1,
        9,
        &[PixelDescriptor::RGB8_SRGB],
        &ConvertOptions::permissive(),
    )
    .unwrap();

    assert!(matches!(adapted, PixelCow::Borrowed(_)));
    assert_eq!(adapted.as_slice().stride(), 9);
}

#[test]
fn known_transfer_difference_converts_instead_of_relabeling() {
    let data = [128, 128, 128];
    let target =
        PixelDescriptor::RGB8_SRGB.with_transfer(zenpixels_convert::TransferFunction::Linear);
    let adapted =
        adapt_for_encode_cow(&data, PixelDescriptor::RGB8_SRGB, 1, 1, 3, &[target]).unwrap();

    assert!(adapted.is_owned());
    assert_ne!(adapted.as_slice().row(0), &data);
}
