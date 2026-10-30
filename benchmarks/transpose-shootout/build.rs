fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var_os("CARGO_FEATURE_CPP_SIMD").is_none() {
        return;
    }
    let manifest = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let simd = manifest.join("third_party/Simd");
    let cmakelists = simd.join("prj/cmake/CMakeLists.txt");
    assert!(
        cmakelists.exists(),
        "third_party/Simd missing — run ./fetch_simd.sh before building with --features cpp-simd"
    );
    // Simd's CMake has no install target we rely on; build the static lib and
    // link straight out of the build dir. SIMD_TEST=OFF skips their test exe.
    let dst = cmake::Config::new(simd.join("prj/cmake"))
        .define("SIMD_TEST", "OFF")
        .define("SIMD_SHARED", "OFF")
        .profile("Release")
        .build_target("Simd")
        .build();
    // The lib lands in <out>/build (single-config generators).
    println!("cargo:rustc-link-search=native={}/build", dst.display());
    println!("cargo:rustc-link-lib=static=Simd");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
