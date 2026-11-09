const assert = require('node:assert/strict');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const url = process.argv[2];
if (!url) throw new Error('Pass the served site URL');
(async () => {
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.goto(url);
    await page.waitForFunction(() => document.getElementById('wasm-status').dataset.loaded === 'true');
    await page.evaluate(() => document.querySelectorAll('details').forEach(e => e.open = true));
    const select = async (id, value) => page.selectOption('#' + id, value);
    const formats = await page.$$eval('#format option', o => o.map(x => x.value));
    const alignments = await page.$$eval('#align option', o => o.map(x => x.value));
    let combinations = 0;
    for (const format of formats) for (const align of alignments) {
      await select('format', format); await select('align', align);
      assert(!await page.locator('body').innerText().then(t => /undefined|NaN/.test(t)), format + '/' + align);
      assert.notEqual(await page.locator('#wasm-status').getAttribute('data-state'), 'error');
      combinations++;
    }
    // Exercise every source/destination route through the actual WASM boundary.
    const matrix = await page.evaluate(() => {
      const counts = { converted: 0, refused: 0 };
      for (const from of FORMATS) for (const to of FORMATS) {
        const cfg = { ...S, format: from.id, be: false, range: 'Full', transfer: 'Srgb', primaries: 'Bt709', alphaMode: 'Straight', w: 1, h: 1, align: 0 };
        const source = compute(cfg);
        const request = { from: converterDescriptor(cfg), to: converterDescriptor({ ...cfg, format: to.id }), width: 1, rows: 1, stride: pixelBytes(source).length, bytes: pixelBytes(source), luma: 'Bt709' };
        const result = JSON.parse(wasmConvert(JSON.stringify(request)));
        if (result.error) counts.refused++;
        else {
          if (result.bytes.length !== CT[to.ct].bytes * to.chans.length || !result.plan) throw new Error('Malformed output: ' + from.id + '/' + to.id);
          counts.converted++;
        }
      }
      return counts;
    });
    await select('format', 'Rgb8'); await select('transfer', 'Srgb'); await select('primaries', 'Bt709');
    await select('range', 'Full'); await select('to-format', 'Bgr8');
    assert.equal(await page.locator('#wasm-status').getAttribute('data-state'), 'match');
    const swizzle = await page.evaluate(() => window.lastWasmConversion);
    assert.deepEqual(swizzle.result.bytes.slice(0, 3), swizzle.request.bytes.slice(0, 3).reverse());
    await select('format', 'Rgb16'); await select('align', '64');
    await page.locator('#w').fill('3'); await page.locator('#w').dispatchEvent('change');
    await select('to-format', 'Rgba16');
    const strided = await page.evaluate(() => window.lastWasmConversion);
    assert.equal(strided.result.bytes.length, strided.request.width * strided.request.rows * 8);
    for (let y = 0; y < strided.request.rows; y++) {
      assert.deepEqual(strided.result.bytes.slice(y * 24, y * 24 + 6), strided.request.bytes.slice(y * strided.request.stride, y * strided.request.stride + 6));
    }
    await select('format', 'RgbF32'); await select('transfer', 'Linear');
    await select('to-format', 'Rgb16'); await select('to-transfer', 'Pq');
    assert.equal(await page.locator('#wasm-status').getAttribute('data-state'), 'refused');
    assert.match(await page.locator('#wasm-status').innerText(), /SdrWhiteRequired/);
    await page.locator('#sdr-white').fill('10000');
    assert.equal(await page.locator('#wasm-status').getAttribute('data-state'), 'different');
    await select('to-transfer', 'Hlg');
    assert.match(await page.locator('#wasm-status').innerText(), /HlgRenderingRequired/);
    await select('primaries', 'Bt2020'); await select('to-primaries', 'Bt2020');
    await page.locator('#hlg-enabled').check();
    assert.equal(await page.locator('#wasm-status').getAttribute('data-state'), 'rendered');
    const hlg = await page.evaluate(() => window.lastWasmConversion);
    assert(hlg.result.bytes.length > 0 && !hlg.result.error);
    await page.locator('#hlg-gamma').fill('4');
    assert.match(await page.locator('#wasm-status').innerText(), /gamma must be/);
    await page.locator('#hlg-gamma').fill('1.2');
    await page.locator('#hlg-enabled').uncheck();
    // Read the SAME interleaved RGB10/alpha1 values with both existing APIs.
    await select('storage-case', 'rgba10a1');
    const padded = await page.evaluate(() => window.lastStorageInspection);
    assert.deepEqual(padded.planeCodes, padded.nativeCodes);
    assert.deepEqual(padded.planeCodes[0], ['1023','1023','1023','1']);
    await page.locator('#storage-padding').uncheck();
    const unpadded = await page.evaluate(() => window.lastStorageInspection);
    assert.deepEqual(unpadded.planeCodes, padded.planeCodes);
    assert.notDeepEqual(unpadded.bytes, padded.bytes);
    await select('storage-offset', '1');
    assert.deepEqual((await page.evaluate(() => window.lastStorageInspection)).nativeCodes, padded.nativeCodes);
    for (const role of ['p010','crcb']) {
      await select('storage-case', role);
      const result = await page.evaluate(() => window.lastStorageInspection);
      assert.deepEqual(result.planeCodes[0], ['1023','1023']);
      assert.equal(result.nativeCodes, null);
      assert.match(result.nativeStatus, /No CbCr/);
    }
    await select('storage-case', 'u64');
    assert.deepEqual((await page.evaluate(() => window.lastStorageInspection)).nativeCodes, [['18446744073709551615']]);
    await select('storage-case', 'i64');
    assert.deepEqual((await page.evaluate(() => window.lastStorageInspection)).nativeCodes, [['-1']]);
    await select('to-transfer', 'Srgb');
    await select('format', 'Cmyk8');
    assert.equal(await page.locator('#wasm-status').getAttribute('data-state'), 'refused');
    assert.match(await page.locator('#wasm-status').innerText(), /NeedsCms/);
    await page.setViewportSize({ width: 390, height: 844 });
    assert(await page.locator('#to-format').isVisible());
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ combinations, formats: formats.length, matrix, pageErrors: errors.length }));
    const review = await page.request.get(new URL('review.html', url.endsWith('/') ? url : url + '/').href);
    assert(review.ok());
    assert.match(await review.text(), /CodePlane::component/);
    const build = await page.request.get(new URL('build-info.json', url.endsWith('/') ? url : url + '/').href);
    assert.match((await build.json()).revision, /^[a-f0-9]{40}$/);
    // Offline/asset-free operation remains functional.
    await page.route('**/wasm/**', route => route.fulfill({ status: 404, body: '' }));
    await page.reload();
    await page.waitForFunction(() => document.getElementById('wasm-status').dataset.state === 'unavailable');
    await page.evaluate(() => document.querySelectorAll('details').forEach(e => e.open = true));
    await select('format', 'Rgb16');
    assert.match(await page.locator('#code').innerText(), /PixelFormat::Rgb16/);
    assert.deepEqual(errors, []);
    console.log('WASM absence fallback passed');
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exit(1); });
