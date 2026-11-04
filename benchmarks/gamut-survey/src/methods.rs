// Research references, not proposed zen source. f64 CSS matrices are from
// zenpixels benchmarks/css-parity at 4322c4dd; Ottosson algorithm port below.
use crate::matrices::*;
pub fn enc(x:f64)->f64 { let a=x.abs(); if a>0.0031308 {x.signum()*(1.055*a.powf(1.0/2.4)-0.055)}else{12.92*x} }
pub fn dec(x:f64)->f64 { let a=x.abs(); if a>0.04045 {x.signum()*((a+0.055)/1.055).powf(2.4)}else{x/12.92} }
pub fn lab(rgb:[f64;3])->[f64;3]{mul(&CSS_LMS_TO_OKLAB,mul(&CSS_XYZ_TO_LMS,mul(&CSS_SRGB_TO_XYZ,rgb)).map(f64::cbrt))}
pub fn unlab(v:[f64;3])->[f64;3]{mul(&inv(&CSS_SRGB_TO_XYZ),mul(&inv(&CSS_XYZ_TO_LMS),mul(&inv(&CSS_LMS_TO_OKLAB),v).map(|x|x*x*x)))}
pub fn de(a:[f64;3],b:[f64;3])->f64{(0..3).map(|i|(a[i]-b[i]).powi(2)).sum::<f64>().sqrt()}
pub fn ing(v:[f64;3])->bool{v.iter().all(|x|x.is_finite() && (0.0..=1.0).contains(x))}
pub fn clip(v:[f64;3])->[f64;3]{v.map(|x|x.clamp(0.0,1.0))}
pub fn hue(l:[f64;3])->f64{l[2].atan2(l[1]).to_degrees()}
pub fn hue_delta(a:[f64;3],b:[f64;3])->f64{(hue(b)-hue(a)+180.0).rem_euclid(360.0)-180.0}
pub fn chroma(l:[f64;3])->f64{l[1].hypot(l[2])}
// Mirrors the requested brief's binary search including encoded tolerance.
pub fn css(linear:[f64;3])->[f64;3]{css_impl(linear,0.0)}
pub fn css_brief(linear:[f64;3])->[f64;3]{css_impl(linear,0.0001)}
fn css_impl(linear:[f64;3],tolerance:f64)->[f64;3]{
 const JND:f64=0.02; const EPS:f64=0.0001;
 let enc3=|c:[f64;3]|c.map(enc);
 let clipped=|c:[f64;3]|clip(enc3(c));
 let origin=lab(linear); let [l,a,b]=origin;
 if l>=1.0{return [1.0;3]} if l<=0.0{return [0.0;3]}
 let in_gamut=|c:[f64;3]|enc3(c).iter().all(|v|(-tolerance..=1.0+tolerance).contains(v));
 if in_gamut(linear){return enc3(linear)}
 let c0=a.hypot(b); let (sh,ch)=b.atan2(a).sin_cos();
 let at=|c:f64|unlab([l,c*ch,c*sh]);
 let(mut min,mut max,mut min_in)=(0.0,c0,true);let mut out=clipped(linear);
 if de(lab(out.map(dec)),origin)<JND{return out}
 while max-min>EPS {
  let c=(min+max)/2.0;let current=at(c);
  if min_in&&in_gamut(current){min=c;continue}
  out=clipped(current);let e=de(lab(out.map(dec)),lab(current));
  if e<JND {if JND-e<EPS{return out}min_in=false;min=c;}else{max=c;}
 }
 out
}
// Port of Bjorn Ottosson, https://bottosson.github.io/posts/gamutclipping/.
// Copyright (c) 2021 Björn Ottosson, MIT. Full license in ottosson-LICENSE.
// Fixed-work polynomial + one Halley refinement; not an unbounded search.
fn ott_lab(v:[f64;3])->[f64;3]{
 let [r,g,b]=v;let l=(0.4122214708*r+0.5363325363*g+0.0514459929*b).cbrt();
 let m=(0.2119034982*r+0.6806995451*g+0.1073969566*b).cbrt();
 let s=(0.0883024619*r+0.2817188376*g+0.6299787005*b).cbrt();
 [0.2104542553*l+0.7936177850*m-0.0040720468*s,1.9779984951*l-2.4285922050*m+0.4505937099*s,0.0259040371*l+0.7827717662*m-0.8086757660*s]
}
const OTT_RGB:M=[[4.0767416621,-3.3077115913,0.2309699292],[-1.2684380046,2.6097574011,-0.3413193965],[-0.0041960863,-0.7034186147,1.7076147010]];
fn ott_unlab(v:[f64;3])->[f64;3]{let [l,a,b]=v;mul(&OTT_RGB,[l+0.3963377774*a+0.2158037573*b,l-0.1055613458*a-0.0638541728*b,l-0.0894841775*a-1.2914855480*b].map(|x|x*x*x))}
fn max_sat(a:f64,b:f64)->f64{
 let(k,w)=if -1.88170328*a-0.80936493*b>1.0 {([1.19086277,1.76576728,0.59662641,0.75515197,0.56771245],OTT_RGB[0])}
 else if 1.81444104*a-1.19445276*b>1.0 {([0.73956515,-0.45954404,0.08285427,0.12541070,0.14503204],OTT_RGB[1])}
 else{([1.35733652,-0.00915799,-1.15130210,-0.50559606,0.00692167],OTT_RGB[2])};
 let mut s=k[0]+k[1]*a+k[2]*b+k[3]*a*a+k[4]*a*b;
 let ks=[0.3963377774*a+0.2158037573*b,-0.1055613458*a-0.0638541728*b,-0.0894841775*a-1.2914855480*b];
 let mut f=0.0;let mut f1=0.0;let mut f2=0.0;
 for i in 0..3{let v=1.0+s*ks[i];f+=w[i]*v*v*v;f1+=w[i]*3.0*ks[i]*v*v;f2+=w[i]*6.0*ks[i]*ks[i]*v;}
 s-=f*f1/(f1*f1-0.5*f*f2);s
}
fn intersect(a:f64,b:f64,l:f64,c:f64,l0:f64)->f64{
 let s=max_sat(a,b);let at=ott_unlab([1.0,s*a,s*b]);let lc=(1.0/at.into_iter().fold(f64::NEG_INFINITY,f64::max)).cbrt();let cc=lc*s;
 if (l-l0)*cc-(lc-l0)*c<=0.0{return cc*l0/(c*lc+cc*(l0-l))}
 let mut t=cc*(l0-1.0)/(c*(lc-1.0)+cc*(l0-l));
 let k=[0.3963377774*a+0.2158037573*b,-0.1055613458*a-0.0638541728*b,-0.0894841775*a-1.2914855480*b];
 let dt=k.map(|v|l-l0+c*v);let ll=l0*(1.0-t)+t*l;let ch=t*c;let v=k.map(|q|ll+ch*q);
 let cube=v.map(|q|q*q*q);let d=core::array::from_fn(|i|3.0*dt[i]*v[i]*v[i]);let d2=core::array::from_fn(|i|6.0*dt[i]*dt[i]*v[i]);
 let vals=mul(&OTT_RGB,cube);let der=mul(&OTT_RGB,d);let der2=mul(&OTT_RGB,d2);
 let mut add=f64::MAX;
 for i in 0..3{let f=vals[i]-1.0;let u=der[i]/(der[i]*der[i]-0.5*f*der2[i]);if u>=0.0{add=add.min(-f*u)}}
 t+=add;t
}
pub fn ott_raw(rgb:[f64;3])->[f64;3]{
 if ing(rgb){return rgb}let [l,a,b]=ott_lab(rgb);let c=a.hypot(b).max(0.00001);let(a,b)=(a/c,b/c);
 let ld=l-0.5;let e=0.5+ld.abs()+0.05*c;
 let l0=0.5*(1.0+ld.signum()*(e-(e*e-2.0*ld.abs()).sqrt()));
 let t=intersect(a,b,l,c,l0);ott_unlab([l0*(1.0-t)+t*l,t*c*a,t*c*b])
}
