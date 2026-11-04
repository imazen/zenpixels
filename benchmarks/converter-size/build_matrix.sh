cd ~/tmp/cs
run(){ name=$1; shift; env "$@" CARGO_TARGET_DIR=~/tmp/cs-m/$name CARGO_PROFILE_RELEASE_DEBUG=0 CARGO_BUILD_JOBS=8 nice -n 19 cargo build --release --bins $T >/dev/null 2>~/tmp/csa/$name.err || { echo "$name BUILD FAILED"; tail -3 ~/tmp/csa/$name.err | cut -c1-100; return; }; d=~/tmp/cs-m/$name/release; [ -n "$T" ] && d=~/tmp/cs-m/$name/${T#--target }/release; e=$S; if [ -n "$S" ]; then b=$d/base.wasm; c=$d/conv.wasm; else strip -o $d/b.s $d/base; strip -o $d/c.s $d/conv; b=$d/b.s; c=$d/c.s; fi; printf '%-34s base=%8d conv=%8d added=%8d\n' $name $(stat -c%s $b) $(stat -c%s $c) $(( $(stat -c%s $c)-$(stat -c%s $b) )); }
T=""; S=""
run o3 CARGO_PROFILE_RELEASE_OPT_LEVEL=3
run o2 CARGO_PROFILE_RELEASE_OPT_LEVEL=2
run os CARGO_PROFILE_RELEASE_OPT_LEVEL=s
run oz CARGO_PROFILE_RELEASE_OPT_LEVEL=z
run o3_cpu_v3 CARGO_PROFILE_RELEASE_OPT_LEVEL=3 RUSTFLAGS="-Ctarget-cpu=x86-64-v3"
run o3_cpu_v4 CARGO_PROFILE_RELEASE_OPT_LEVEL=3 RUSTFLAGS="-Ctarget-cpu=x86-64-v4"
run os_cpu_v3 CARGO_PROFILE_RELEASE_OPT_LEVEL=s RUSTFLAGS="-Ctarget-cpu=x86-64-v3"
T="--target wasm32-wasip1"; S=wasm
run wasm_o3 CARGO_PROFILE_RELEASE_OPT_LEVEL=3
run wasm_o3_simd CARGO_PROFILE_RELEASE_OPT_LEVEL=3 RUSTFLAGS="-Ctarget-feature=+simd128"
run wasm_os_simd CARGO_PROFILE_RELEASE_OPT_LEVEL=s RUSTFLAGS="-Ctarget-feature=+simd128"
run wasm_oz_simd CARGO_PROFILE_RELEASE_OPT_LEVEL=z RUSTFLAGS="-Ctarget-feature=+simd128"
