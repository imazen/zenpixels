const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const url = process.argv[2];
if (!url) throw new Error('Pass the served site URL');
(async () => {
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
    const errors = [];
    const screenshots = process.env.EXPLORER_SCREENSHOTS;
    if (screenshots) fs.mkdirSync(screenshots); // A fresh, explicitly requested capture directory.
    page.on('pageerror', e => errors.push(e.message));
    await page.goto(url);
    await page.waitForFunction(() => {
      const state = document.getElementById('wasm-status');
      return state.dataset.loaded === 'true' || state.dataset.state === 'unavailable';
    });
    assert.equal(await page.locator('#wasm-status').getAttribute('data-loaded'), 'true', await page.locator('#wasm-status').innerText());
    await page.evaluate(() => document.querySelectorAll('details').forEach(e => e.open = true));
    const select = async (id, value) => page.selectOption('#' + id, value);
    const formats = await page.$$eval('#format option', o => o.map(x => x.value));
    const alignments = await page.$$eval('#align option', o => o.map(x => x.value));
    let combinations = 0;
    assert.deepEqual(
      (await page.evaluate(() => window.formatCatalog.map(f => f.id))).sort(),
      [...formats].sort(),
    );
    const rustExamples = [];
    for (const format of formats) for (const align of alignments) {
      await select('format', format); await select('align', align);
      assert(!await page.locator('body').innerText().then(t => /undefined|NaN/.test(t)), format + '/' + align);
      assert.notEqual(await page.locator('#wasm-status').getAttribute('data-state'), 'error');
      rustExamples.push(await page.locator('#code').innerText());
      combinations++;
    }
    // Unknown declarations must be explicit in the copied code too.
    await select('format', 'Rgb8');
    await select('transfer', 'Unknown'); await select('primaries', 'Unknown');
    assert.match(await page.locator('#code').innerText(), /with_transfer\(TransferFunction::Unknown\)/);
    assert.match(await page.locator('#code').innerText(), /with_primaries\(ColorPrimaries::Unknown\)/);
    rustExamples.push(await page.locator('#code').innerText());
    await select('format', 'RgbaI16'); await select('alphamode', 'Opaque');
    assert.match(await page.locator('#code').innerText(), /with_alpha\(Some\(AlphaMode::Opaque\)\)/);
    rustExamples.push(await page.locator('#code').innerText());
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
    // Existing component readers handle interleaved RGB10/alpha1 without another view.
    await select('storage-case', 'rgba10a1');
    const padded = await page.evaluate(() => window.lastStorageInspection);
    assert.deepEqual(padded.planeCodes[0], ['1023','1023','1023','1']);
    await page.locator('#storage-padding').uncheck();
    const unpadded = await page.evaluate(() => window.lastStorageInspection);
    assert.deepEqual(unpadded.planeCodes, padded.planeCodes);
    assert.notDeepEqual(unpadded.bytes, padded.bytes);
    for (const role of ['p010','crcb']) {
      await select('storage-case', role);
      const result = await page.evaluate(() => window.lastStorageInspection);
      assert.deepEqual(result.planeCodes[0], ['1023','1023']);
    }
    await select('storage-case', 'rgb565');
    const rgb565 = await page.evaluate(() => window.lastStorageInspection);
    assert.deepEqual(rgb565.planeCodes[0], ['31','63','31']);
    assert.deepEqual(rgb565.canonicalBytes.slice(0, 3), [255,255,255]);
    assert(await page.locator('#storage-padding').isDisabled());
    assert(await page.locator('#storage-canonical').isVisible());
    await select('to-transfer', 'Srgb');
    await select('format', 'Cmyk8');
    assert.equal(await page.locator('#wasm-status').getAttribute('data-state'), 'refused');
    assert.match(await page.locator('#wasm-status').innerText(), /NeedsCms/);
    // Wide native words and open advanced panels exposed the original overflow.
    await select('format', 'RgbaF64');
    for (const width of [320, 360, 390, 768, 1024, 1440]) {
      await page.setViewportSize({ width, height: 844 });
      const extent = await page.evaluate(() => ({page: document.documentElement.scrollWidth, viewport: innerWidth}));
      assert.equal(extent.page, extent.viewport, 'Explorer overflow at ' + width);
      assert(await page.locator('#to-format').isVisible());
    }
    if (screenshots) {
      await select('format', 'Rgba8'); await select('transfer', 'Srgb');
      await select('primaries', 'Bt709'); await select('alphamode', 'Straight');
      await page.evaluate(() => document.querySelectorAll('details').forEach(e => e.open = false));
      for (const width of [390, 1440]) {
        await page.setViewportSize({ width, height: 1000 });
        await page.evaluate(() => scrollTo(0, 0));
        await page.screenshot({ path: path.join(screenshots, `explorer-${width}.png`) });
      }
    }
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ combinations, formats: formats.length, matrix, pageErrors: errors.length }));
    const review = await page.request.get(new URL('review.html', url.endsWith('/') ? url : url + '/').href);
    assert(review.ok());
    assert.match(await review.text(), /CodePlane::component/);
    const build = await page.request.get(new URL('build-info.json', url.endsWith('/') ? url : url + '/').href);
    assert.match((await build.json()).revision, /^[a-f0-9]{40}$/);
    // Every linked teaching page must be readable without document overflow.
    const extraRustExamples = [];
    for (const teachingPath of ['guide.html', 'speed-and-size.html', 'eli5.html', 'transfers.html', 'color-hdr.html', 'kernels.html', 'color-explorer/', 'review.html', 'format_browser.html']) {
      const teaching = await browser.newPage();
      const pageErrors = [];
      teaching.on('pageerror', e => pageErrors.push(e.message));
      await teaching.goto(new URL(teachingPath, url.endsWith('/') ? url : url + '/').href);
      if (teachingPath === 'guide.html') {
        assert.match(await teaching.locator('body').innerText(), /Borrow the part you need/);
        assert.match(await teaching.locator('body').innerText(), /parent stride/);
        assert.match(await teaching.locator('body').innerText(), /Balance speed and light compilation/);
      }
      if (teachingPath === 'speed-and-size.html') {
        assert.match(await teaching.locator('body').innerText(), /Accelerators::WEB_SDR/);
        assert.match(await teaching.locator('body').innerText(), /enable_all/);
      }
      if (teachingPath === 'eli5.html') {
        assert.match(await teaching.locator('body').innerText(), /Plain-English Mental Model/);
        assert.match(await teaching.locator('body').innerText(), /FloatFacts/);
      }
      if (teachingPath === 'transfers.html') {
        assert((await teaching.locator('#svg-eotf polyline').count()) >= 5, 'transfers.html EOTF polylines missing');
        assert((await teaching.locator('#svg-density polyline').count()) >= 5, 'transfers.html density polylines missing');
        assert.equal(await teaching.locator('#hlg-coupled-grid .hlg-cell').count(), 4, 'transfers.html HLG coupled grid missing');
        // Verify Linear actually goes over 1.0 (e.g. probe near upper right -> >1000 nits)
        await teaching.locator('#probe-x').fill('0.85');
        await teaching.locator('#probe-x').dispatchEvent('input');
        assert.match(await teaching.locator('#probe-readout').innerText(), /Linear:\s*[1-9]\d{2,3}\./, 'Linear should exceed SDR white (>1.0) when unclamped');
        // Test OETF direction + rule checkboxes (allow >1, allow <0, odd-mirror, narrow range, log/lin, nits/relative, HLG OOTF, Chart 2 X axis)
        for (const dir of ['oetf', 'eotf']) {
          await teaching.locator(`.dir-btn[data-dir="${dir}"]`).click();
          assert((await teaching.locator('#svg-eotf polyline').count()) >= 5, `transfers.html dir=${dir}`);
        }
        await teaching.locator('#rule-neg').check();
        assert((await teaching.locator('#svg-eotf polyline').count()) >= 5, 'transfers.html allow <0');
        await teaching.locator('#rule-mirror').uncheck();
        await teaching.locator('#rule-mirror').check();
        await teaching.locator('#rule-narrow').check();
        assert((await teaching.locator('#svg-eotf rect').count()) >= 1, 'transfers.html narrow shading');
        await teaching.locator('#rule-narrow').uncheck();
        await teaching.locator('#rule-over1').uncheck();
        assert((await teaching.locator('#svg-eotf polyline').count()) >= 5, 'transfers.html clamped <=1');
        await teaching.locator('#rule-over1').check();
        await teaching.locator('#rule-log-light').check();
        await teaching.locator('#rule-log-val').check();
        await teaching.locator('#rule-nits').uncheck();
        await teaching.locator('#rule-nits').check();
        await teaching.locator('#rule-hlg-ootf').uncheck();
        await teaching.locator('#rule-hlg-ootf').check();
        await teaching.locator('#rule-dens-nits').uncheck();
        await teaching.locator('#rule-dens-nits').check();
        await teaching.selectOption('#bit-depth', '8');
        await teaching.selectOption('#bit-depth', '10');
        await teaching.locator('.hlg-pbtn[data-lw="2000"]').click();
        assert.equal(await teaching.locator('#hlg-peak-val').innerText(), '2000');
        await teaching.locator('.hlg-pbtn[data-lw="1000"]').click();
        const combos = [
          { sf: 'RGB8', stf: 'Srgb', sp: 'Bt709', df: 'RGBF32', dtf: 'Linear', dp: 'Bt709', gp: 'Clip', dpol: 'Round', ap: 'web_sdr' },
          { sf: 'RGB8', stf: 'Srgb', sp: 'DisplayP3', df: 'OKLABF32', dtf: 'Linear', dp: 'Bt709', gp: 'Clip', dpol: 'Round', ap: 'custom' },
          { sf: 'RGBF32', stf: 'Linear', sp: 'Bt2020', df: 'RGB16', dtf: 'Pq', dp: 'Bt2020', gp: 'Clip', dpol: 'Round', ap: 'default' },
          { sf: 'RGB16', stf: 'Pq', sp: 'Bt2020', df: 'RGB8', dtf: 'Srgb', dp: 'Bt709', gp: 'Clip', dpol: 'Round', ap: 'all' },
          { sf: 'RGB16', stf: 'Hlg', sp: 'Bt2020', df: 'RGB8', dtf: 'Srgb', dp: 'Bt709', gp: 'Clip', dpol: 'Round', ap: 'enable_all' },
          { sf: 'RGBA8', stf: 'Gamma22', sp: 'AdobeRgb', df: 'RGB8', dtf: 'Srgb', dp: 'Bt709', gp: 'SoftCompress', dpol: 'Truncate', ap: 'web_sdr' },
        ];
        for (const c of combos) {
          await teaching.selectOption('#src-fmt', c.sf);
          if (c.sf !== 'OKLABF32') { await teaching.selectOption('#src-tf', c.stf); await teaching.selectOption('#src-prim', c.sp); }
          await teaching.selectOption('#dst-fmt', c.df);
          if (c.df !== 'OKLABF32') { await teaching.selectOption('#dst-tf', c.dtf); await teaching.selectOption('#dst-prim', c.dp); }
          await teaching.selectOption('#gamut-pol', c.gp);
          await teaching.selectOption('#depth-pol', c.dpol);
          await teaching.selectOption('#accel-pol', c.ap);
          extraRustExamples.push(await teaching.evaluate(() => window.getTransferRustCode()));
        }
        // Also verify custom HlgRendering (2000 nit HDR TV) + HLG -> PQ codegen compiles and runs!
        await teaching.locator('.hlg-pbtn[data-lw="2000"]').click();
        await teaching.selectOption('#src-fmt', 'RGB16');
        await teaching.selectOption('#src-tf', 'Hlg');
        await teaching.selectOption('#dst-fmt', 'RGB16');
        await teaching.selectOption('#dst-tf', 'Pq');
        await teaching.selectOption('#dst-prim', 'Bt2020');
        await teaching.selectOption('#gamut-pol', 'Clip');
        await teaching.selectOption('#depth-pol', 'Round');
        const hlgPqCode = await teaching.evaluate(() => window.getTransferRustCode());
        assert.match(hlgPqCode, /HlgRendering::new\(Nits::new\(2000\.0\)/);
        extraRustExamples.push(hlgPqCode);
      }
      if (teachingPath === 'color-hdr.html') {
        assert((await teaching.locator('#svg-cie polygon').count()) >= 5, 'color-hdr.html CIE polygons missing');
        assert((await teaching.locator('#svg-ruler rect').count()) >= 1, 'color-hdr.html HDR ruler missing');
        assert((await teaching.locator('#svg-tm polyline').count()) >= 2, 'color-hdr.html BT.2446A curve missing');
        await teaching.locator('.diag-btn[data-proj="uv"]').click();
        assert((await teaching.locator('#svg-cie polygon').count()) >= 5);
        await teaching.selectOption('#cicp-preset', 'BT2100_PQ');
        assert.match(await teaching.locator('#cicp-detail').innerText(), /NonIdentityMatrix/);
        const scenarios = await teaching.evaluate(() => Object.values(window.HDR_SCENARIOS));
        assert(scenarios.length >= 6);
        extraRustExamples.push(...scenarios);
      }
      if (teachingPath === 'kernels.html') {
        const allCount = await teaching.locator('#klist details.kitem').count();
        assert(allCount >= 150, `kernels.html expected >=150 extracted kernels, got ${allCount}`);
        await teaching.locator('.pbtn[data-preset="WEB_SDR"]').click();
        const webCount = await teaching.locator('#klist details.kitem').count();
        assert(webCount > 10 && webCount < allCount, `WEB_SDR filter expected 10 < ${webCount} < ${allCount}`);
        await teaching.locator('.pbtn[data-preset="SCALAR"]').click();
        const scalarCount = await teaching.locator('#klist details.kitem').count();
        assert(scalarCount >= 1 && scalarCount < webCount, `SCALAR filter expected 1 <= ${scalarCount} < ${webCount}`);
        await teaching.locator('.pbtn[data-preset="any"]').click();
      }
      assert.deepEqual(pageErrors, [], teachingPath + ' page errors: ' + pageErrors.join('; '));
      for (const width of [320, 390, 768, 1440]) {
        await teaching.setViewportSize({ width, height: 900 });
        const extent = await teaching.evaluate(() => ({page: document.documentElement.scrollWidth, viewport: innerWidth}));
        assert.equal(extent.page, extent.viewport, teachingPath + ' overflow at ' + width);
      }
      if (screenshots && teachingPath === 'guide.html') {
        for (const width of [390, 1440]) {
          await teaching.setViewportSize({ width, height: 1000 });
          await teaching.screenshot({ path: path.join(screenshots, `guide-${width}.png`) });
        }
        await teaching.setViewportSize({ width: 390, height: 1000 });
        await teaching.locator('#borrow-the-part-you-need').scrollIntoViewIfNeeded();
        await teaching.screenshot({ path: path.join(screenshots, 'guide-crop.png') });
      }
      await teaching.close();
    }
    if (process.env.EXPLORER_RUST_EXAMPLES) {
      const allExamples = [...rustExamples, ...extraRustExamples];
      fs.writeFileSync(process.env.EXPLORER_RUST_EXAMPLES,
        allExamples.map((s, i) => `## Explorer example ${i}\n\n\`\`\`rust\n# fn main() -> Result<(), Box<dyn std::error::Error>> {\n${s}\n# Ok(())\n# }\n\`\`\`\n`).join('\n'));
    }
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
