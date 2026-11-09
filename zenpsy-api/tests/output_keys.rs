//! Public validation of small and large native output-key sets.
//! Map witnesses are synthetic one-cell views, requested only in these tests.
use zenpsy_api::{
    zenpixels::{PixelDescriptor, PixelSlice, plane::*},
    *,
};

static PRIMARY: ScoreSpec = ScoreSpec::new("primary", Direction::Lower);
static CELL: [f32; 1] = [0.0];
struct Native {
    keys: Vec<ScoreSpec>,
    maps: bool,
}
impl Evaluation for Native {
    fn scores(&self) -> impl Iterator<Item = Score<'_>> {
        std::iter::once(Score::new(&PRIMARY, 0.0))
            .filter(|_| self.maps)
            .chain(
                self.keys
                    .iter()
                    .filter(|_| !self.maps)
                    .map(|s| Score::new(s, 0.0)),
            )
    }
    fn maps(&self) -> impl Iterator<Item = MapView<'_>> {
        self.keys.iter().filter(|_| self.maps).map(|s| {
            MapView::new(
                s,
                AnyPlane::F32(PlaneRef::new(&CELL[..], 1, 1, 1).unwrap()),
                MapGrid::new(1, 1).unwrap(),
            )
        })
    }
}
struct MetricFixture {
    names: Vec<String>,
    maps: bool,
}
impl Metric for MetricFixture {
    type Output = Native;
    fn id(&self) -> MetricId {
        MetricId::new("key-control", Implementation::new("test", "1", 1))
    }
    fn evaluate(
        &mut self,
        _: &PixelSlice<'_>,
        _: &PixelSlice<'_>,
        _: EvalOpts<'_>,
    ) -> MetricResult<Native> {
        Ok(Native {
            keys: self
                .names
                .iter()
                .map(|k| ScoreSpec::new("native", Direction::Lower).with_key(k.clone()))
                .collect(),
            maps: self.maps,
        })
    }
}
fn names(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("native-{i}")).collect()
}
fn run(keys: Vec<String>, maps: bool) -> MetricResult<ErasedEvaluation> {
    let mut m = Erased::new(Erased::new(MetricFixture { names: keys, maps }));
    let pixels = PixelSlice::new(&[0], 1, 1, 1, PixelDescriptor::GRAY8).unwrap();
    score(
        &mut m,
        &pixels,
        &pixels,
        if maps {
            EvalOpts::new().with_native_map()
        } else {
            EvalOpts::new()
        },
    )
}
#[test]
fn native_names_validate_small_and_large_score_and_map_sets() {
    for maps in [false, true] {
        for n in [64, 65, 130] {
            let output = run(names(n), maps).unwrap();
            assert_eq!(
                if maps {
                    output.maps().count()
                } else {
                    output.scores().count()
                },
                n
            );
        }
    }
}
#[test]
fn empty_and_duplicate_names_are_refused_early_and_late() {
    for maps in [false, true] {
        for (at, duplicate_of) in [(1, 0), (64, 0), (129, 0), (129, 64)] {
            let mut keys = names(130);
            keys[at] = keys[duplicate_of].clone();
            let error = run(keys, maps).err().unwrap();
            assert_eq!(error.error().kind(), ErrorKind::Internal);
            assert_eq!(
                error.error().message(),
                Some("empty or duplicate output key")
            );
        }
        for at in [0, 64, 129] {
            let mut keys = names(130);
            keys[at].clear();
            assert_eq!(
                run(keys, maps).err().unwrap().error().kind(),
                ErrorKind::Internal
            );
        }
    }
}
