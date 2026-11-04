// Row-0 alignment that plain Vec allocations get from the system allocator.
fn main() {
    println!("bytes      u8%64 u16%64 f32%64 (5 allocations each, mod 64 / mod 4096)");
    for &bytes in &[64usize, 1 << 10, 16 << 10, 127 << 10, 128 << 10, 1 << 20, 32 << 20] {
        let mut keep: Vec<Box<dyn std::any::Any>> = Vec::new();
        let mut line = format!("{bytes:>9}");
        let mut u8s = Vec::new(); let mut u16s = Vec::new(); let mut f32s = Vec::new();
        for _ in 0..5 {
            let a = vec![0u8; bytes]; let b = vec![0u16; bytes / 2]; let c = vec![0f32; bytes / 4];
            u8s.push((a.as_ptr() as usize % 64, a.as_ptr() as usize % 4096));
            u16s.push((b.as_ptr() as usize % 64, b.as_ptr() as usize % 4096));
            f32s.push((c.as_ptr() as usize % 64, c.as_ptr() as usize % 4096));
            keep.push(Box::new(a)); keep.push(Box::new(b)); keep.push(Box::new(c));
        }
        line += &format!("  {u8s:?}\n{:>9}  {u16s:?}\n{:>9}  {f32s:?}", "", "");
        println!("{line}");
    }
}
