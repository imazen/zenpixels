#!/usr/bin/env node
// Bounded ICC/cICP capability witnesses, not a display or tone-mapping oracle.
// PLAYWRIGHT_MODULE may name an existing Playwright installation.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const zlib = require('node:zlib');
const playwright = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

function hash(data) { return crypto.createHash('sha256').update(data).digest('hex'); }
function u32(n) { const b = Buffer.alloc(4); b.writeUInt32BE(n); return b; }
function crc32(data) {
  let crc = 0xffffffff;
  for (const byte of data) {
    crc ^= byte;
    for (let i = 0; i < 8; i++) crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
  }
  return (crc ^ 0xffffffff) >>> 0;
}
function chunk(type, data) {
  const body = Buffer.concat([Buffer.from(type), data]);
  return Buffer.concat([u32(data.length), body, u32(crc32(body))]);
}

// Deliberately contradictory fallback: a large visual witness for whether the
// CMM uses DToB or matrix/TRC. These profiles are diagnostics, never exports.
function rewriteProfile(source, { removeFloat = false, linearize = false, removeCicp = true } = {}) {
  const count = source.readUInt32BE(128);
  const entries = [];
  for (let i = 0; i < count; i++) {
    const pos = 132 + i * 12;
    const tag = source.toString('ascii', pos, pos + 4);
    const offset = source.readUInt32BE(pos + 4), size = source.readUInt32BE(pos + 8);
    if (offset + size > source.length) throw Error('ICC tag outside file');
    if (removeFloat && /^[DB]2[BD][0-3]$/.test(tag)) continue;
    if (removeCicp && tag === 'cicp') continue;
    let data = source.subarray(offset, offset + size);
    if (linearize && ['rTRC', 'gTRC', 'bTRC'].includes(tag)) {
      data = Buffer.alloc(16); data.write('para'); data.writeInt32BE(65536, 12);
    }
    entries.push({ tag, data });
  }
  const header = Buffer.from(source.subarray(0, 128));
  header.fill(0, 84, 100); // Invalidated profile ID, not a claimed original ID.
  const table = Buffer.alloc(4 + entries.length * 12);
  table.writeUInt32BE(entries.length);
  let offset = 128 + table.length;
  const payloads = [];
  entries.forEach(({ tag, data }, i) => {
    table.write(tag, 4 + i * 12); table.writeUInt32BE(offset, 8 + i * 12);
    table.writeUInt32BE(data.length, 12 + i * 12);
    const padded = Buffer.alloc((data.length + 3) & ~3); data.copy(padded);
    payloads.push(padded); offset += padded.length;
  });
  header.writeUInt32BE(offset);
  return Buffer.concat([header, table, ...payloads]);
}

function png(samples, icc, cicp) {
  const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(samples.length);
  ihdr.writeUInt32BE(1, 4); ihdr[8] = 16; ihdr[9] = 2;
  const row = Buffer.alloc(1 + samples.length * 6);
  samples.forEach((rgb, x) => rgb.forEach((v, c) => row.writeUInt16BE(v, 1 + x * 6 + c * 2)));
  const parts = [Buffer.from('89504e470d0a1a0a', 'hex'), chunk('IHDR', ihdr)];
  if (cicp) parts.push(chunk('cICP', Buffer.from([9, 16, 0, 1])));
  if (icc) parts.push(chunk('iCCP', Buffer.concat([Buffer.from('PQ capability witness\0\0'), zlib.deflateSync(icc)])));
  parts.push(chunk('IDAT', zlib.deflateSync(row)), chunk('IEND', Buffer.alloc(0)));
  return Buffer.concat(parts);
}

function compare(a, b) {
  if (a.length !== b.length) throw Error('output shape mismatch');
  let different = 0, maxCodeDifference = 0;
  a.forEach((v, i) => {
    if (i % 4 === 3) return;
    different += Number(v !== b[i]); maxCodeDifference = Math.max(maxCodeDifference, Math.abs(v - b[i]));
  });
  return { differentRgbChannels: different, maxCodeDifference };
}

async function main() {
  const [originalPath, floatPath, externalPath, outputPath, ...requested] = process.argv.slice(2);
  if (!outputPath) throw Error('expected ORIGINAL.icc FLOAT.icc EXTERNAL.icc FRESH_OUTPUT_DIR [chromium firefox firefox-cms webkit]');
  const output = path.resolve(outputPath); fs.mkdirSync(output); // Never replace prior evidence.
  const original = fs.readFileSync(originalPath), floating = fs.readFileSync(floatPath);
  const external = fs.readFileSync(externalPath);
  const fallback = rewriteProfile(floating, { linearize: true });
  const linear = rewriteProfile(floating, { linearize: true, removeFloat: true });
  const originalWithoutCicp = rewriteProfile(original), floatWithoutCicp = rewriteProfile(floating);
  const samples = [];
  for (let i = 0; i <= 32; i++) samples.push(Array(3).fill(Math.round(i * 65535 / 32)));
  for (const r of [0, 32768, 65535]) for (const g of [0, 32768, 65535]) for (const b of [0, 32768, 65535]) samples.push([r, g, b]);
  const variants = {
    original: [originalWithoutCicp, false], float: [floatWithoutCicp, false],
    'original-embedded-cicp': [original, false],
    'float-linear-fallback': [fallback, false], 'linear-only': [linear, false],
    'float-cicp': [floatWithoutCicp, true], 'linear-cicp': [linear, true],
    'cicp-only': [null, true],
    external: [external, false], untagged: [null, false],
  };
  const fixtures = {};
  for (const [name, [icc, cicp]] of Object.entries(variants)) {
    const data = png(samples, icc, cicp); fs.writeFileSync(path.join(output, `${name}.png`), data);
    fixtures[name] = { pngSha256: hash(data), iccSha256: icc ? hash(icc) : null, cicp };
    if (icc) fs.writeFileSync(path.join(output, `${name}.icc`), icc);
  }
  const report = {
    scope: 'PNG16 to canvas sRGB U8; capability witnesses, no HDR display accuracy claim',
    playwright: require(path.join(path.dirname(require.resolve(process.env.PLAYWRIGHT_MODULE || 'playwright')), 'package.json')).version,
    sources: { original: hash(original), floating: hash(floating), external: hash(external) },
    width: samples.length, height: 1, samples, fixtures, readers: {},
  };
  let failures = 0;
  for (const name of requested.length ? requested : ['chromium', 'firefox', 'firefox-cms', 'webkit']) {
    let browser;
    try {
      const engine = name === 'firefox-cms' ? 'firefox' : name;
      if (!playwright[engine]) throw Error(`unknown browser ${name}`);
      const options = name === 'firefox-cms' ? {
        firefoxUserPrefs: { 'gfx.color_management.mode': 1, 'gfx.color_management.enablev4': true, 'gfx.color_management.force_srgb': true },
      } : {};
      // The bundled WPE launcher replaces LD_LIBRARY_PATH. A direct launch
      // permits isolated runtime libraries without changing the host install.
      const webkitBundle = engine === 'webkit' && process.env.WEBKIT_PROBE_BUNDLE;
      if (webkitBundle) {
        options.executablePath = path.join(webkitBundle, 'bin/MiniBrowser');
        options.env = { ...process.env,
          WEBKIT_EXEC_PATH: path.join(webkitBundle, 'bin'),
          WEBKIT_INJECTED_BUNDLE_PATH: path.join(webkitBundle, 'lib'),
          WEBKIT_INSPECTOR_RESOURCES_PATH: path.join(webkitBundle, 'share'),
          LD_LIBRARY_PATH: [path.join(webkitBundle, 'lib'), path.join(webkitBundle, 'sys/lib'), process.env.LD_LIBRARY_PATH || ''].join(':'),
        };
      }
      browser = await playwright[engine].launch(options);
      const page = await browser.newPage();
      const values = {};
      for (const variant of Object.keys(variants)) {
        const url = `data:image/png;base64,${fs.readFileSync(path.join(output, `${variant}.png`)).toString('base64')}`;
        values[variant] = await page.evaluate(async ({ url, width }) => {
          const img = new Image(); img.src = url; await img.decode();
          if (img.naturalWidth !== width || img.naturalHeight !== 1) throw Error('decoded shape mismatch');
          const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = 1;
          const ctx = canvas.getContext('2d', { colorSpace: 'srgb' }); ctx.drawImage(img, 0, 0);
          return Array.from(ctx.getImageData(0, 0, width, 1, { colorSpace: 'srgb' }).data);
        }, { url, width: samples.length });
      }
      report.readers[name] = {
        version: browser.version(), executableSha256: hash(fs.readFileSync(options.executablePath || playwright[engine].executablePath())),
        options: { ...(options.firefoxUserPrefs ? { firefoxUserPrefs: options.firefoxUserPrefs } : {}), isolatedWebkitRuntime: Boolean(webkitBundle) }, values,
        comparisons: Object.fromEntries([
          ['float-vs-original', 'float', 'original'],
          ['float-vs-diagnostic-fallback', 'float', 'float-linear-fallback'],
          ['diagnostic-vs-linear-only', 'float-linear-fallback', 'linear-only'],
          ['cicp-overrides-icc-witness', 'float-cicp', 'linear-cicp'],
          ['dual-vs-cicp-only', 'float-cicp', 'cicp-only'],
          ['cicp-changes-linear-icc', 'linear-cicp', 'linear-only'],
          ['linear-icc-vs-untagged', 'linear-only', 'untagged'],
          ['embedded-cicp-vs-icc-curves', 'original-embedded-cicp', 'original'],
        ].map(([label, a, b]) => [label, compare(values[a], values[b])])),
      };
      console.log(name, browser.version(), JSON.stringify(report.readers[name].comparisons));
    } catch (error) {
      failures++; report.readers[name] = { error: String(error) }; console.error(name, error);
    } finally { if (browser) await browser.close(); }
    fs.writeFileSync(path.join(output, 'report.json'), JSON.stringify(report, null, 2) + '\n');
  }
  if (failures) throw Error(`${failures} reader(s) failed; evidence preserved`);
}
main().catch(error => { console.error(error); process.exitCode = 1; });
