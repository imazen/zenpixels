// Writes the reference table for tests/css_color_oracle.rs. Color.js is the oracle only.
// node css_oracle.mjs <path to Color.js src/index.js> > ../../zenpixels-convert/tests/data/css_oracle.tsv
const { default: Color } = await import(process.argv[2]);
const cases = [
  '#f00', '#f008', '#112233', '#11223344', 'red', 'rebeccapurple', 'aliceblue', 'transparent',
  'rgb(255 0 0)', 'rgb(255,128,0)', 'rgba(1,2,3,0.5)', 'rgb(100% 50% 25%)', 'rgb(12.5 40 80)',
  'rgb(300 -20 40)', 'rgb(10 20 30 / 25%)',
  'hsl(120 100% 50%)', 'hsl(240,100%,50%)', 'hsl(30deg 50% 40%)', 'hsl(0.5turn 80% 30% / 0.5)',
  'hsl(200grad 60% 60%)', 'hsl(3.14rad 60% 60%)', 'hsla(10,20%,30%,0.4)', 'hsl(-60 50% 50%)',
  'hwb(120 0% 0%)', 'hwb(30 20% 30%)', 'hwb(0 60% 60%)', 'hwb(200 10% 20% / 0.7)',
  'color(srgb 1 0.5 0)', 'color(srgb 110% -10% 50%)', 'color(srgb-linear 0.5 0.25 1)',
  'color(display-p3 1 0 0)', 'color(display-p3 0.2 0.9 0.3 / 0.5)', 'color(display-p3 none 0.5 0.5)',
  'color(a98-rgb 0.2 0.9 0.3)', 'color(rec2020 1 0 0)', 'color(rec2020 0.1 0.5 0.9)',
  'color(rec2020 -0.1 1.2 0.5)', 'color(prophoto-rgb 0.4 0.9 0.2)', 'color(prophoto-rgb 0.02 0.01 0.03)',
  'color(xyz 0.2 0.3 0.4)', 'color(xyz-d65 0.5 0.5 0.5)', 'color(xyz-d50 0.2 0.3 0.4)',
  'lab(50% 40 60)', 'lab(29.2345 39.3825 20.0664)', 'lab(80 -100 80)', 'lab(50% 50% -50%)',
  'lab(5 10 -10)', 'lab(100 0 0)', 'lab(0 0 0)', 'lab(52 40 60 / 0.5)',
  'lch(50% 60 200)', 'lch(29.2345 44.2 27)', 'lch(70 150 120)', 'lch(70 50% 0.25turn)',
  'oklab(60% 0.1 -0.1)', 'oklab(0.5 0.2 0.1)', 'oklab(40% 50% -50%)', 'oklab(0.9 0 0)', 'oklab(1 0 0)',
  'oklch(70% 0.2 150)', 'oklch(0.4 0.3 30)', 'oklch(0.7 50% 270deg)', 'oklch(0.6 0.1 none)',
  'oklch(0.5 0.15 0.5turn / 0.25)',
];
for (const css of cases) {
  const color = new Color(css);
  const [r, g, b] = color.to('rec2020-linear').coords;
  const a = color.alpha?.valueOf() ?? 1;
  console.log([css, r, g, b, a].join('\t'));
}
