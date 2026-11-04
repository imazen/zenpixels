#![forbid(unsafe_code)]
#![allow(dead_code)]
#[path = "../matrices.rs"] mod matrices;
#[path = "../methods.rs"] mod methods;
fn main(){
 let x:Vec<f64>=std::env::args().skip(1).map(|x|x.parse().unwrap()).collect();
 let mut rgb=[x.first().copied().unwrap_or(1.2),x.get(1).copied().unwrap_or(-0.1),x.get(2).copied().unwrap_or(0.5)];
 #[cfg(feature="size-css")]{rgb=methods::css(std::hint::black_box(rgb));}
 #[cfg(feature="size-ott")]{rgb=methods::clip(methods::ott_raw(std::hint::black_box(rgb))).map(methods::enc);}
 #[cfg(feature="size-soft")]{
  use zenpixels_convert::{hdr::SoftCompress,oklab};
  let p=zenpixels::ColorPrimaries::Bt709;
  let s=SoftCompress::from_matrices(&oklab::rgb_to_lms_matrix(p).unwrap(),&oklab::lms_to_rgb_matrix(p).unwrap(),0.96);
  let mut row=[rgb.map(|v|v as f32)];s.apply_strip(&mut row);rgb=row[0].map(|v|methods::enc(v.clamp(0.0,1.0)as f64));
 }
 #[cfg(not(any(feature="size-css",feature="size-ott",feature="size-soft")))]{rgb=methods::clip(std::hint::black_box(rgb)).map(methods::enc);}
 println!("{:?}",std::hint::black_box(rgb));
}
