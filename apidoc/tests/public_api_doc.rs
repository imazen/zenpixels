//! Public-API surface snapshots for the PARENT workspace (docs/public-api/).
//! Shared implementation + format docs: the `zenutils-apidoc` crate.
//!
//! `no_file_meta_header()` + `no_autotraits_summary()` strip two
//! perpetually-churning lines from the rendered snapshots: the `# files:`
//! line-count header at the top of `<crate>.txt`, and the `X types implement
//! all of: …` summary at the top of each `## auto traits` block. Both
//! counters shift whenever the API grows by a few lines, producing
//! diff-noise that conveys no semver signal. The explicit
//! `Type: !Trait …` auto-trait exception lines stay — those are the
//! actual signal we want to see in PR diffs.
#[test]
fn public_api_surface_docs_are_current() {
    zenutils_apidoc::ApiDoc::new()
        .workspace_dir("..")
        .no_file_meta_header()
        .no_autotraits_summary()
        .run();
}
