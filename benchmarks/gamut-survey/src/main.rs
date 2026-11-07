#![forbid(unsafe_code)]
mod matrices; mod methods;
use matrices::*;use methods::*;
use std::{fs,io::{Write,BufWriter},sync::Arc};
use zenpixels::ColorPrimaries as P;
use zenpixels_convert::{oklab, hdr::SoftCompress};
type Xf=Arc<dyn moxcms::TransformExecutor<f32>+Send+Sync>;
type Xl=Arc<dyn moxlocal::TransformExecutor<f32>+Send+Sync>;
enum Kind{Clip,Css,CssBrief,Ott,Soft(SoftCompress),Film,Mox(Xf),Local(Xl)}
struct Method{name:String,matrix:M,kind:Kind}
fn matrix(g:usize)->M{let p=if g==0{[(0.68,0.32),(0.265,0.69),(0.15,0.06)]}else{[(0.708,0.292),(0.17,0.797),(0.131,0.046)]};matmul(&inv(&CSS_SRGB_TO_XYZ),&rgb_to_xyz(p))}
fn make_methods(g:usize)->Vec<Method>{
 let matrix=matrix(g);let mut out=Vec::new();
 for(name,kind)in[("clip-f64",Kind::Clip),("css-search-f64",Kind::Css),("css-brief-tolerance-f64",Kind::CssBrief),("ottosson-adaptive-f64",Kind::Ott),("softcompress-f32-k096",Kind::Soft(SoftCompress::new(&oklab::OklabSpace::for_primaries(P::Bt709).unwrap(),0.96))),("mox-filmlike-f32",Kind::Film)]{out.push(Method{name:name.into(),matrix,kind})}
 let mut src=if g==0{moxcms::ColorProfile::new_display_p3()}else{moxcms::ColorProfile::new_bt2020()};src.cicp=None;
 for v in [&mut src.red_trc,&mut src.green_trc,&mut src.blue_trc]{*v=Some(moxcms::ToneReprCurve::Parametric(vec![1.0]));}
 let dst=moxcms::ColorProfile::new_srgb();
 for (name,intent) in [("perceptual",moxcms::RenderingIntent::Perceptual),("relative",moxcms::RenderingIntent::RelativeColorimetric),("saturation",moxcms::RenderingIntent::Saturation),("absolute",moxcms::RenderingIntent::AbsoluteColorimetric)]{
  let o=moxcms::TransformOptions{rendering_intent:intent,allow_use_cicp_transfer:false,interpolation_method:moxcms::InterpolationMethod::Tetrahedral,barycentric_weight_scale:moxcms::BarycentricWeightScale::High,..Default::default()};
  out.push(Method{name:format!("mox091-{name}"),matrix,kind:Kind::Mox(src.create_transform_f32(moxcms::Layout::Rgb,&dst,moxcms::Layout::Rgb,o).unwrap())});
 }
 let o=moxcms::TransformOptions{allow_extended_range_rgb_xyz:true,prefer_fixed_point:false,allow_use_cicp_transfer:false,..Default::default()};
 out.push(Method{name:"mox091-extended-clipped".into(),matrix,kind:Kind::Mox(src.create_transform_f32(moxcms::Layout::Rgb,&dst,moxcms::Layout::Rgb,o).unwrap())});
 let mut src=if g==0{moxlocal::ColorProfile::new_display_p3()}else{moxlocal::ColorProfile::new_bt2020()};src.cicp=None;
 for v in [&mut src.red_trc,&mut src.green_trc,&mut src.blue_trc]{*v=Some(moxlocal::ToneReprCurve::Parametric(vec![1.0]));}
 let dst=moxlocal::ColorProfile::new_srgb();
 for (name,intent) in [("perceptual",moxlocal::RenderingIntent::Perceptual),("relative",moxlocal::RenderingIntent::RelativeColorimetric),("saturation",moxlocal::RenderingIntent::Saturation),("absolute",moxlocal::RenderingIntent::AbsoluteColorimetric)]{
  let o=moxlocal::TransformOptions{rendering_intent:intent,allow_use_cicp_transfer:false,interpolation_method:moxlocal::InterpolationMethod::Tetrahedral,barycentric_weight_scale:moxlocal::BarycentricWeightScale::High,..Default::default()};
  out.push(Method{name:format!("mox-local090-{name}"),matrix,kind:Kind::Local(src.create_transform_f32(moxlocal::Layout::Rgb,&dst,moxlocal::Layout::Rgb,o).unwrap())});
 }
 for n in [17,33]{let (method,bytes)=make_clut(g,n);println!("synthetic CLUT gamut={g} grid={n} encoded_profile_bytes={bytes}");out.push(method);}
 out
}

fn make_clut(g:usize,n:usize)->(Method,usize){
 use moxlocal::{ColorProfile,LutDataType,LutStore,LutType,LutWarehouse,Matrix3d};
 let m=matrix(g);let dst=ColorProfile::new_srgb();let xyz=dst.rgb_to_xyz_matrix().v;
 let mut clut=Vec::with_capacity(n*n*n*3);
 for r in 0..n{for gg in 0..n{for b in 0..n{let s=[r as f64/(n-1)as f64,gg as f64/(n-1)as f64,b as f64/(n-1)as f64];let mapped=css(mul(&m,s)).map(dec);let pcs=mul(&xyz,mapped);for v in pcs{clut.push((v/(65535.0/32768.0)*65535.0).round().clamp(0.0,65535.0)as u16);}}}}
 let lut=LutWarehouse::Lut(LutDataType{num_input_channels:3,num_output_channels:3,num_clut_grid_points:n as u8,matrix:Matrix3d::IDENTITY,num_input_table_entries:2,num_output_table_entries:2,input_table:LutStore::Store16(vec![0,65535,0,65535,0,65535]),output_table:LutStore::Store16(vec![0,65535,0,65535,0,65535]),clut_table:LutStore::Store16(clut),lut_type:LutType::Lut16});
 let mut src=if g==0{ColorProfile::new_display_p3()}else{ColorProfile::new_bt2020()};src.cicp=None;src.lut_a_to_b_perceptual=Some(lut);
 let encoded=src.encode().unwrap();let bytes=encoded.len();let parsed=ColorProfile::new_from_slice(&encoded).unwrap();
 let options=moxlocal::TransformOptions{rendering_intent:moxlocal::RenderingIntent::Perceptual,allow_use_cicp_transfer:false,interpolation_method:moxlocal::InterpolationMethod::Tetrahedral,barycentric_weight_scale:moxlocal::BarycentricWeightScale::High,..Default::default()};
 let tr=parsed.create_transform_f32(moxlocal::Layout::Rgb,&dst,moxlocal::Layout::Rgb,options).unwrap();
 println!("missing-intent synthetic grid={n}: {:?}",parsed.create_transform_f32(moxlocal::Layout::Rgb,&dst,moxlocal::Layout::Rgb,moxlocal::TransformOptions{rendering_intent:moxlocal::RenderingIntent::RelativeColorimetric,..options}).err());
 (Method{name:format!("synthetic-css-clut-{n}"),matrix:m,kind:Kind::Local(tr)},bytes)
}

impl Method{
 fn run(&self,src:&[[f64;3]],src32:&[f32],dst:&mut[[f64;3]],dst32:&mut[f32]){
  match &self.kind {
   Kind::Mox(t)=>{t.transform(src32,dst32).unwrap();if self.name.ends_with("clipped"){for x in dst32{*x=x.clamp(0.0,1.0)}}},
   Kind::Local(t)=>t.transform(src32,dst32).unwrap(),
   Kind::Soft(t)=>{
    let row=dst32.as_chunks_mut::<3>().0;
    for (d,&s) in row.iter_mut().zip(src){*d=mul(&self.matrix,s).map(|v|v as f32);}
    t.apply_strip(row);
    for(d,x)in dst.iter_mut().zip(row){*d=x.map(|v|enc(v.clamp(0.0,1.0) as f64));}
   },
   _=>for(d,&x)in dst.iter_mut().zip(src){let l=mul(&self.matrix,x);*d=match &self.kind{
    Kind::Clip=>clip(l).map(enc),Kind::Css=>css(l),Kind::CssBrief=>css_brief(l),Kind::Ott=>clip(ott_raw(l)).map(enc),
    Kind::Film=>{let x=moxcms::filmlike_clip(moxcms::Rgb::new(l[0] as f32,l[1] as f32,l[2] as f32));clip([x.r as f64,x.g as f64,x.b as f64]).map(enc)},_=>unreachable!()};}
  }
 }
 fn eval(&self,src:&[[f64;3]])->Vec<[f64;3]>{let s:Vec<f32>=src.iter().flatten().map(|&x|x as f32).collect();let mut d=vec![[0.0;3];src.len()];let mut df=vec![0.0;s.len()];self.run(src,&s,&mut d,&mut df);if matches!(self.kind,Kind::Mox(_)|Kind::Local(_)){for(o,i)in d.iter_mut().zip(df.chunks_exact(3)){*o=[i[0] as f64,i[1] as f64,i[2] as f64];}}d}
}
fn output(path:&str)->BufWriter<fs::File>{BufWriter::new(fs::File::create(path).unwrap())}
fn quant(rgb:[f64;3])->[u8;3]{rgb.map(|x|(x.clamp(0.0,1.0)*255.0).round() as u8)}
fn measure(){
 let mut prim=output("primaries.csv");writeln!(prim,"gamut,method,primary,r8,g8,b8,hue_float_deg,de_float,delta_L,hue_u8_deg,de_u8").unwrap();
 let mut sweep=output("sweep.csv");writeln!(sweep,"gamut,method,n,in_gamut_n,in_gamut_changed_gt_1e-6,in_gamut_max_de,mean_de,max_de,mean_abs_hue,max_abs_hue,max_abs_dL,outside_1e-7,nonfinite").unwrap();
 let mut ramps=output("ramps.csv");writeln!(ramps,"gamut,method,ramps,steps,chroma_reversals_1e-7,max_chroma_reverse,ramps_with_reverse,max_adjacent_de,u8_equal_adjacent,total_adjacent").unwrap();
 let mut detail=output("ramp_failures.csv");writeln!(detail,"gamut,method,L,hue,input_C,prev_out_C,out_C,adjacent_de").unwrap();
 let mut residual=output("residual.csv");writeln!(residual,"gamut,method,n,raw_outside_1e-7,max_excursion").unwrap();
 for g in 0..2{
 let gn=if g==0{"P3"}else{"BT2020"};let m=matrix(g);let mi=inv(&m);let ms=make_methods(g);
 let cube:Vec<_>=(0..17).flat_map(|r|(0..17).flat_map(move|gg|(0..17).map(move|b|[r as f64/16.0,gg as f64/16.0,b as f64/16.0]))).collect();
 let all_mox:Vec<_>=ms.iter().filter(|x|x.name.starts_with("mox091-")&&!x.name.contains("extended")||x.name.starts_with("mox-local090-")).map(|x|x.eval(&cube)).collect();for row in &all_mox[1..]{assert_eq!(row,&all_mox[0],"all intents / both mox versions on cube");}println!("exact mox equality {gn}: {} colors x 8 intent/version combinations",cube.len());
 let reference:Vec<_>=cube.iter().map(|&x|lab(mul(&m,x))).collect();
 for method in &ms{
 let out=method.eval(&[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]);
 for(i,&rgb)in out.iter().enumerate(){let orig=lab(mul(&m,core::array::from_fn(|j|if i==j{1.0}else{0.0})));let actual=lab(rgb.map(dec));let q=quant(rgb);let ql=lab(q.map(|x|dec(x as f64/255.0)));writeln!(prim,"{gn},{},{},{},{},{},{:.9},{:.9},{:.9},{:.9},{:.9}",method.name,["red","green","blue"][i],q[0],q[1],q[2],hue_delta(orig,actual),de(orig,actual),actual[0]-orig[0],hue_delta(orig,ql),de(orig,ql)).unwrap();}
 let out=method.eval(&cube);let(mut ni,mut changed,mut imax,mut sum,mut max,mut hs,mut hm,mut hn,mut lm,mut outside,mut nf)=(0,0,0.0_f64,0.0_f64,0.0_f64,0.0_f64,0.0_f64,0,0.0_f64,0,0);
 for(i,&o)in out.iter().enumerate(){if !o.iter().all(|v|v.is_finite()){nf+=1;continue}if o.iter().any(|&v|!(-1e-7..=1.0+1e-7).contains(&v)){outside+=1}let actual=lab(o.map(dec));let e=de(reference[i],actual);sum+=e;max=max.max(e);lm=lm.max((reference[i][0]-actual[0]).abs());if chroma(reference[i])>1e-4 && chroma(actual)>1e-4{let h=hue_delta(reference[i],actual).abs();hs+=h;hm=hm.max(h);hn+=1;}
 let linear=mul(&m,cube[i]);if linear.iter().all(|v|(-1e-12..=1.0+1e-12).contains(v)){ni+=1;imax=imax.max(e);if e>1e-6{changed+=1;}}
 }
 writeln!(sweep,"{gn},{},{},{ni},{changed},{imax:.10},{:.10},{max:.10},{:.8},{hm:.8},{lm:.10},{outside},{nf}",method.name,cube.len(),sum/cube.len() as f64,hs/hn as f64).unwrap();
 if matches!(method.kind,Kind::Ott|Kind::Soft(_)){
 let(mut count,mut excursion)=(0,0.0_f64);for &x in &cube{let l=mul(&m,x);let raw=match &method.kind{Kind::Ott=>ott_raw(l),Kind::Soft(t)=>{let mut x=[l.map(|v|v as f32)];t.apply_strip(&mut x);x[0].map(|v|v as f64)},_=>unreachable!()};let e=raw.iter().map(|&v|(-v).max(v-1.0).max(0.0)).fold(0.0,f64::max);if e>1e-7{count+=1;}excursion=excursion.max(e);}
 writeln!(residual,"{gn},{},{},{count},{excursion:.12}",method.name,cube.len()).unwrap();
 }
 }
 let mut totals=vec![(0_u64,0.0_f64,0_u64,0.0_f64,0_u64);ms.len()];
 for ll in [0.2,0.4,0.6,0.8]{for h in (0..360).step_by(5){let(sh,ch)=(h as f64).to_radians().sin_cos();let(mut lo,mut hi)=(0.0,1.0);for _ in 0..44{let c=(lo+hi)/2.0;if ing(mul(&mi,unlab([ll,c*ch,c*sh]))){lo=c}else{hi=c}}
 let src:Vec<_>=(0..1024).map(|i|{let c=lo*i as f64/1023.0;mul(&mi,unlab([ll,c*ch,c*sh]))}).collect();
 for(k,method)in ms.iter().enumerate(){let out=method.eval(&src);let mut prev=lab(out[0].map(dec));let mut prevq=quant(out[0]);let mut has=false;for(i,&o)in out.iter().enumerate().skip(1){let ol=lab(o.map(dec));let reversal=chroma(prev)-chroma(ol);let step=de(prev,ol);let q=quant(o);let t=&mut totals[k];if reversal>1e-7{t.0+=1;has=true;if t.0<=20{writeln!(detail,"{gn},{},{ll},{h},{:.10},{:.10},{:.10},{step:.10}",method.name,lo*i as f64/1023.0,chroma(prev),chroma(ol)).unwrap();}}t.1=t.1.max(reversal);t.3=t.3.max(step);if q==prevq{t.4+=1}prev=ol;prevq=q;}if has{totals[k].2+=1}}
 }}
 for(method,t)in ms.iter().zip(totals){writeln!(ramps,"{gn},{},288,1024,{},{:.10},{},{:.10},{},294624",method.name,t.0,t.1,t.2,t.3,t.4).unwrap();}
 println!("finished quality {gn}");
 }
 probes();
}
fn probes(){
 let p=moxcms::ColorProfile::new_display_p3();let dst=moxcms::ColorProfile::new_srgb();let s=[-0.1,0.5,1.2];
 for ext in [false,true]{let t=p.create_transform_f32(moxcms::Layout::Rgb,&dst,moxcms::Layout::Rgb,moxcms::TransformOptions{allow_extended_range_rgb_xyz:ext,prefer_fixed_point:!ext,allow_use_cicp_transfer:false,..Default::default()}).unwrap();let mut d=[0.;3];t.transform(&s,&mut d).unwrap();println!("extended probe ext={ext} input={s:?} output={d:?}")}
 for x in [[2.,0.,0.],[2.,2.,2.],[-0.5,1.2,0.2]]{let o=moxcms::filmlike_clip(moxcms::Rgb::new(x[0],x[1],x[2]));println!("filmlike probe {x:?} -> {:?}",[o.r,o.g,o.b]);}
}
fn benchmark(){
 let result=zenbench::run(|suite|{for g in 0..2{let gn=if g==0{"P3"}else{"BT2020"};suite.group(format!("{gn}-1024-linear-to-encoded"),|group|{
 let cfg=group.config();cfg.max_rounds=30;cfg.min_rounds=20;cfg.auto_rounds=false;cfg.warmup_time=std::time::Duration::from_millis(200);cfg.max_time=std::time::Duration::from_secs(3);cfg.max_wall_time=std::time::Duration::from_secs(50);cfg.min_iterations=1;cfg.max_iterations=64;
 group.throughput(zenbench::Throughput::Elements(1024));
 for method in make_methods(g){let name=method.name.clone();let src:Vec<_>=(0..1024).map(|i|{let r=(i*313+17)%1024;let g=(i*619+71)%1024;let b=(i*109+337)%1024;[r as f64/1023.0,g as f64/1023.0,b as f64/1023.0]}).collect();let src32:Vec<_>=src.iter().flatten().map(|&x|x as f32).collect();let mut dst=vec![[0.;3];1024];let mut dst32=vec![0.;3072];group.bench(name,move|b|b.iter(||{method.run(std::hint::black_box(&src),std::hint::black_box(&src32),&mut dst,&mut dst32);std::hint::black_box(&dst);std::hint::black_box(&dst32);}));}
 });}
 suite.group("native-row1024",|group|{
 let cfg=group.config();cfg.max_rounds=30;cfg.min_rounds=20;cfg.auto_rounds=false;cfg.warmup_time=std::time::Duration::from_millis(200);cfg.max_time=std::time::Duration::from_secs(3);cfg.max_iterations=64;
 group.throughput(zenbench::Throughput::Elements(1024));
 for (gn,p) in [("P3",P::DisplayP3),("BT2020",P::Bt2020)]{
 let mut converter=zenpixels_convert::RowConverter::new(zenpixels::PixelDescriptor::RGBF32_LINEAR.with_primaries(p),zenpixels::PixelDescriptor::RGBF32_LINEAR.with_transfer(zenpixels::TransferFunction::Srgb)).unwrap();
 let src:Vec<f32>=(0..1024).flat_map(|i|[((i*313+17)%1024)as f32/1023.0,((i*619+71)%1024)as f32/1023.0,((i*109+337)%1024)as f32/1023.0]).collect();let mut dst=vec![0_f32;3072];
 group.bench(format!("zen-row-{gn}-f32"),move|b|b.iter(||{converter.convert_row(bytemuck::cast_slice(std::hint::black_box(&src)),bytemuck::cast_slice_mut(&mut dst),1024);std::hint::black_box(&dst);}));
 }
 let p=P::Bt709;let soft=SoftCompress::new(&oklab::OklabSpace::for_primaries(p).unwrap(),0.96);let src:Vec<_>=(0..1024).map(|i|[(i%1024)as f32/1023.0,-0.1,0.3]).collect();let mut dst=src.clone();
 group.bench("soft-kernel-copy-1024",move|b|b.iter(||{dst.copy_from_slice(std::hint::black_box(&src));soft.apply_strip(&mut dst);std::hint::black_box(&dst);}));
 });});
 fs::write("bench.json",serde_json::to_string_pretty(&result).unwrap()).unwrap();println!("{}",result.to_markdown());
}
fn validation(){
 let mut rows=Vec::new();
 let mut accuracy=output("clut-accuracy.csv");writeln!(accuracy,"gamut,method,n,mean_de_from_css,max_de_from_css,max_encoded_channel_error").unwrap();
 for g in 0..2{
 let src:Vec<_>=(0..1024).map(|i|[((i*313+17)%1024)as f64/1023.0,((i*619+71)%1024)as f64/1023.0,((i*109+337)%1024)as f64/1023.0]).collect();
 let refs:Vec<_>=src.iter().map(|&x|css(mul(&matrix(g),x))).collect();
 for method in make_methods(g).iter().filter(|m|m.name.starts_with("synthetic")){let out=method.eval(&src);let(mut sum,mut max,mut ce)=(0.0_f64,0.0_f64,0.0_f64);for (&a,&b)in out.iter().zip(&refs){let e=de(lab(a.map(dec)),lab(b.map(dec)));sum+=e;max=max.max(e);for i in 0..3{ce=ce.max((a[i]-b[i]).abs());}}writeln!(accuracy,"{g},{},1024,{},{max},{ce}",method.name,sum/1024.0).unwrap();}
 for i in 0..1024{let rgb=[((i*313+17)%1024)as f64/1023.0,((i*619+71)%1024)as f64/1023.0,((i*109+337)%1024)as f64/1023.0];let lin=mul(&matrix(g),rgb);rows.push(serde_json::json!({"linear":lin,"css":css(lin),"ott":ott_raw(lin)}));}}
 fs::write("validation-input.json",serde_json::to_string(&rows).unwrap()).unwrap();
}
fn main(){match std::env::args().nth(1).as_deref(){Some("validation")=>validation(),Some("measure")=>measure(),Some("bench")=>benchmark(),Some("probes")=>probes(),_=>panic!("use measure / bench / probes")}}
