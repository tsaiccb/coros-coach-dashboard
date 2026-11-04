/* ============================================================================
 * 跑姿分析 (Running Form Analysis) — form.js
 * In-browser video gait analysis for the Louis Tsai run-training dashboard.
 *  - MediaPipe Pose Landmarker (tasks-vision) runs locally in the browser; the
 *    video never leaves the device.
 *  - Kinematic engine: view detection, body-scale calibration, gait events
 *    (foot-plateau + Zeni 2008 pelvis-relative method), cadence, contact/flight
 *    time, vertical oscillation, sagittal joint angles, frontal-plane pelvis /
 *    knee / crossover metrics, L-R asymmetry, Morin (2005) spring-mass stiffness.
 *  - COROS matching: the clip's recording timestamp (MP4/MOV metadata) is aligned
 *    to the per-second COROS dynamics (dyn/<labelId>.json) of the same workout.
 *  - Scoring + coaching expert system oriented to half-marathon running economy.
 * Pure-analysis functions are exposed on window.FORM so they can be unit-tested
 * in Node with synthetic kinematics (test_form.js).
 * ========================================================================== */
(function (global) {
  "use strict";

  // ---------------------------------------------------------------- constants
  const LM = { nose: 0, l_eye: 2, r_eye: 5, l_ear: 7, r_ear: 8, l_sh: 11, r_sh: 12, l_el: 13, r_el: 14, l_wr: 15, r_wr: 16,
    l_hip: 23, r_hip: 24, l_knee: 25, r_knee: 26, l_ank: 27, r_ank: 28, l_heel: 29, r_heel: 30, l_toe: 31, r_toe: 32 };
  const SIDES = { L: { sh: 11, el: 13, wr: 15, hip: 23, knee: 25, ank: 27, heel: 29, toe: 31, ear: 7 }, R: { sh: 12, el: 14, wr: 16, hip: 24, knee: 26, ank: 28, heel: 30, toe: 32, ear: 8 } };
  const CONNECTIONS = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28], [27, 29], [29, 31], [27, 31], [28, 30], [30, 32], [28, 32], [0, 7], [0, 8]];
  // Drillis & Contini segment lengths as a fraction of stature
  const SEG = { thigh: 0.245, shank: 0.246, trunk: 0.288, leg: 0.53 };
  const G = 9.81;
  const DEG = 180 / Math.PI;

  // ---------------------------------------------------------------- math utils
  const isNum = (v) => typeof v === "number" && Number.isFinite(v);
  const mean = (a) => { const x = a.filter(isNum); return x.length ? x.reduce((s, v) => s + v, 0) / x.length : null; };
  const median = (a) => { const x = a.filter(isNum).slice().sort((p, q) => p - q); if (!x.length) return null; const m = x.length >> 1; return x.length % 2 ? x[m] : (x[m - 1] + x[m]) / 2; };
  const pctl = (a, p) => { const x = a.filter(isNum).slice().sort((p1, q) => p1 - q); if (!x.length) return null; const i = (x.length - 1) * p; const lo = Math.floor(i), hi = Math.ceil(i); return x[lo] + (x[hi] - x[lo]) * (i - lo); };
  const sd = (a) => { const x = a.filter(isNum); if (x.length < 2) return null; const m = mean(x); return Math.sqrt(x.reduce((s, v) => s + (v - m) * (v - m), 0) / (x.length - 1)); };
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const r1 = (v) => (isNum(v) ? Math.round(v * 10) / 10 : null);
  const r0 = (v) => (isNum(v) ? Math.round(v) : null);
  const lerp = (a, b, f) => a + (b - a) * f;
  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  /** interior angle (deg) at b between ba and bc */
  const angleAt = (a, b, c) => { const v1x = a.x - b.x, v1y = a.y - b.y, v2x = c.x - b.x, v2y = c.y - b.y; const d = Math.hypot(v1x, v1y) * Math.hypot(v2x, v2y); if (!d) return null; return Math.acos(clamp((v1x * v2x + v1y * v2y) / d, -1, 1)) * DEG; };

  function movingAverage(a, w) {
    w = Math.max(1, Math.round(w)); if (w <= 1) return a.slice();
    const out = new Array(a.length); const h = Math.floor(w / 2);
    for (let i = 0; i < a.length; i++) { let s = 0, n = 0; for (let j = Math.max(0, i - h); j <= Math.min(a.length - 1, i + h); j++) { if (isNum(a[j])) { s += a[j]; n++; } } out[i] = n ? s / n : null; }
    return out;
  }
  function median3(a) { const out = a.slice(); for (let i = 1; i < a.length - 1; i++) { const x = a[i - 1], y = a[i], z = a[i + 1]; if (isNum(x) && isNum(y) && isNum(z)) out[i] = Math.max(Math.min(x, y), Math.min(Math.max(x, y), z)); } return out; }
  /** fill null gaps by linear interpolation (and edge hold) */
  function fillGaps(a) {
    const out = a.slice(); let i = 0;
    while (i < out.length && !isNum(out[i])) i++;
    if (i === out.length) return out;
    for (let k = 0; k < i; k++) out[k] = out[i];
    let last = i;
    for (let k = i + 1; k < out.length; k++) {
      if (isNum(out[k])) { if (k - last > 1) for (let j = last + 1; j < k; j++) out[j] = lerp(out[last], out[k], (j - last) / (k - last)); last = k; }
    }
    for (let k = last + 1; k < out.length; k++) out[k] = out[last];
    return out;
  }
  /** indices of local maxima with minimum spacing and prominence */
  function localMaxima(sig, minDist, minProm) {
    const idx = [];
    for (let i = 1; i < sig.length - 1; i++) {
      if (!(isNum(sig[i]) && sig[i] >= sig[i - 1] && sig[i] > sig[i + 1])) continue;
      // prominence: drop to the lowest point between this peak and the nearest higher point on each side (bounded window)
      let lo1 = sig[i], lo2 = sig[i];
      for (let j = i - 1; j >= Math.max(0, i - 3 * minDist); j--) { if (sig[j] > sig[i]) break; lo1 = Math.min(lo1, sig[j]); }
      for (let j = i + 1; j <= Math.min(sig.length - 1, i + 3 * minDist); j++) { if (sig[j] > sig[i]) break; lo2 = Math.min(lo2, sig[j]); }
      if (sig[i] - Math.max(lo1, lo2) < minProm) continue;
      if (idx.length && i - idx[idx.length - 1] < minDist) { if (sig[i] > sig[idx[idx.length - 1]]) idx[idx.length - 1] = i; continue; }
      idx.push(i);
    }
    return idx;
  }
  const localMinima = (sig, minDist, minProm) => localMaxima(sig.map((v) => (isNum(v) ? -v : v)), minDist, minProm);
  /** sub-sample extremum refinement (parabolic) -> fractional index */
  function refinePeak(sig, i) { if (i <= 0 || i >= sig.length - 1) return i; const a = sig[i - 1], b = sig[i], c = sig[i + 1]; const d = a - 2 * b + c; if (!isNum(d) || Math.abs(d) < 1e-9) return i; return clamp(i + 0.5 * (a - c) / d, i - 0.5, i + 0.5); }
  const tAt = (t, fi) => { const i = Math.floor(fi); if (i >= t.length - 1) return t[t.length - 1]; return lerp(t[i], t[i + 1], fi - i); };
  /** autocorrelation peak -> fundamental period (in samples) within [minLag,maxLag] */
  function dominantPeriod(sig, minLag, maxLag) {
    const x = sig.filter(isNum); const n = x.length; if (n < minLag * 2) return null;
    const m = mean(x); const z = x.map((v) => v - m); const denom = z.reduce((s, v) => s + v * v, 0) || 1;
    const ac = []; for (let lag = minLag; lag <= Math.min(maxLag, n - 2); lag++) { let s = 0; for (let i = 0; i + lag < n; i++) s += z[i] * z[i + lag]; ac.push({ lag, r: s / denom * n / (n - lag) }); }
    if (!ac.length) return null;
    // first strong local maximum (fundamental), else global max
    let best = null;
    for (let k = 1; k < ac.length - 1; k++) { if (ac[k].r > ac[k - 1].r && ac[k].r >= ac[k + 1].r && ac[k].r > 0.35) { best = ac[k]; break; } }
    if (!best) best = ac.reduce((p, q) => (q.r > p.r ? q : p));
    if (best.r < 0.2) return null;
    // parabolic refine over lag
    const k = ac.indexOf(best); let lag = best.lag;
    if (k > 0 && k < ac.length - 1) { const a = ac[k - 1].r, b = ac[k].r, c = ac[k + 1].r; const d = a - 2 * b + c; if (Math.abs(d) > 1e-9) lag += 0.5 * (a - c) / d; }
    return { period: lag, r: best.r };
  }

  // ------------------------------------------------------------ preprocessing
  /**
   * frames: [{t: seconds (video time), lm: [33 × {x,y,z,visibility}] normalized}], W/H pixels.
   * Returns per-landmark pixel series P[k] = {x:[], y:[], z:[], v:[]} (gaps filled, median-3 filtered) + quality.
   */
  function preprocess(frames, W, H) {
    const n = frames.length; const P = [];
    for (let k = 0; k < 33; k++) P.push({ x: new Array(n).fill(null), y: new Array(n).fill(null), z: new Array(n).fill(null), v: new Array(n).fill(0) });
    const key = [23, 24, 25, 26, 27, 28];
    let good = 0; const okFrame = new Array(n).fill(false);
    for (let i = 0; i < n; i++) {
      const lm = frames[i].lm; if (!lm || lm.length < 33) continue;
      const vis = mean(key.map((k) => lm[k].visibility ?? 1));
      okFrame[i] = vis >= 0.5; if (okFrame[i]) good++;
      for (let k = 0; k < 33; k++) { const p = lm[k]; P[k].v[i] = p.visibility ?? 1; if (okFrame[i] && (p.visibility ?? 1) >= 0.3) { P[k].x[i] = p.x * W; P[k].y[i] = p.y * H; P[k].z[i] = p.z; } }
    }
    for (let k = 0; k < 33; k++) { P[k].x = median3(fillGaps(P[k].x)); P[k].y = median3(fillGaps(P[k].y)); P[k].z = fillGaps(P[k].z); }
    return { P, quality: n ? good / n : 0, okFrame };
  }
  const pt = (P, k, i) => ({ x: P[k].x[i], y: P[k].y[i], z: P[k].z[i] });
  const mid = (P, a, b, i) => ({ x: (P[a].x[i] + P[b].x[i]) / 2, y: (P[a].y[i] + P[b].y[i]) / 2 });

  // ----------------------------------------------------------- view detection
  function detectView(P, n) {
    const shSep = [], hipSep = [], trunk = [], noseVis = [], lMinusR = [], zL = [], zR = [], toeDir = [];
    for (let i = 0; i < n; i++) {
      shSep.push(Math.abs(P[11].x[i] - P[12].x[i])); hipSep.push(Math.abs(P[23].x[i] - P[24].x[i]));
      trunk.push(dist(mid(P, 11, 12, i), mid(P, 23, 24, i)));
      noseVis.push(P[0].v[i]); lMinusR.push(P[11].x[i] - P[12].x[i]);
      zL.push(mean([P[11].z[i], P[23].z[i], P[25].z[i], P[27].z[i]])); zR.push(mean([P[12].z[i], P[24].z[i], P[26].z[i], P[28].z[i]]));
      toeDir.push(Math.sign((P[31].x[i] - P[29].x[i]) + (P[32].x[i] - P[30].x[i])));
    }
    const ratio = median(shSep) / (median(trunk) || 1), hipRatio = median(hipSep) / (median(trunk) || 1);
    let view;
    if (ratio < 0.42 && hipRatio < 0.32) view = "side"; else if (ratio > 0.55) view = median(lMinusR) > 0 ? "front" : "back"; else view = "oblique";
    const dir = Math.sign(mean(toeDir) || 1) || 1; // +1 = running towards image right
    const nearSide = median(zL) < median(zR) ? "L" : "R";
    return { view, ratio: r1(ratio * 100) / 100, dir, nearSide, noseVis: r1(mean(noseVis) * 100) / 100 };
  }

  // --------------------------------------------------------- scale (px per cm)
  function bodyScale(P, n, heightCm, view) {
    const th = [], sh = [];
    for (let i = 0; i < n; i++) for (const s of ["L", "R"]) { const S = SIDES[s]; th.push(dist(pt(P, S.hip, i), pt(P, S.knee, i))); sh.push(dist(pt(P, S.knee, i), pt(P, S.ank, i))); }
    const thighPx = pctl(th, 0.95), shankPx = pctl(sh, 0.95);
    const est = [thighPx / (SEG.thigh * heightCm), shankPx / (SEG.shank * heightCm)];
    if (view === "side") { const tr = []; for (let i = 0; i < n; i++) tr.push(dist(mid(P, 11, 12, i), mid(P, 23, 24, i))); est.push(pctl(tr, 0.9) / (SEG.trunk * heightCm)); }
    return { pxPerCm: mean(est), thighPx, shankPx };
  }

  // ------------------------------------------------------------- gait events
  /**
   * Foot-plateau contact detection, per foot. footLow = lowest point of the foot (max of heel/toe y).
   * A stance = run of frames where footLow is within `thrPx` of the local ground level (per cycle max).
   * In side view the Zeni (2008) extrema (heel most forward / toe most backward relative to the pelvis)
   * bound each cycle and serve as fallback.
   */
  function detectEvents(P, t, side, opts) {
    const { stridePeriod, pxPerCm, dir, view } = opts; const S = SIDES[side]; const n = t.length;
    const footLow = new Array(n), relHeel = new Array(n), relToe = new Array(n);
    for (let i = 0; i < n; i++) {
      footLow[i] = Math.max(P[S.heel].y[i], P[S.toe].y[i], P[S.ank].y[i]);
      const px = (P[23].x[i] + P[24].x[i]) / 2;
      relHeel[i] = dir * (P[S.heel].x[i] - px); relToe[i] = dir * (P[S.toe].x[i] - px);
    }
    const minDist = Math.max(3, Math.round(stridePeriod * 0.6));
    const promFoot = Math.max(2, 2.5 * pxPerCm);
    // cycle anchors: foot vertical maxima (lowest position) are present in every view
    const lows = localMaxima(footLow, minDist, promFoot);
    const thrPx = Math.max(2, 1.8 * pxPerCm);
    const events = [];
    for (let c = 0; c < lows.length; c++) {
      const i0 = lows[c];
      // cycle window: up to 0.75 stride either side of the anchor, never into the neighbouring stance of the same foot
      const lo = Math.max(c > 0 ? lows[c - 1] + 1 : 0, i0 - Math.round(stridePeriod * 0.75));
      const hi = Math.min(c < lows.length - 1 ? lows[c + 1] - 1 : n - 1, i0 + Math.round(stridePeriod * 0.75));
      let ground = -Infinity; for (let i = Math.max(lo, i0 - Math.round(stridePeriod * 0.4)); i <= Math.min(hi, i0 + Math.round(stridePeriod * 0.4)); i++) ground = Math.max(ground, footLow[i]);
      // plateau around i0
      let a = i0, b = i0;
      while (a > lo && footLow[a - 1] >= ground - thrPx) a--;
      while (b < hi && footLow[b + 1] >= ground - thrPx) b++;
      // sub-frame timing: extrapolate the foot's approach / departure velocity to the ground level
      // (a fixed height threshold would bias IC early and TO late by threshold / foot speed)
      let fa = a, fb = b;
      if (a > lo) {
        const y0 = footLow[a - 1], y1 = footLow[a]; if (y1 !== y0) fa = a - 1 + clamp((ground - thrPx - y0) / (y1 - y0), 0, 1);
        if (a - 2 >= lo) { const v = footLow[a - 1] - footLow[a - 2]; if (v > 0.3 * thrPx) fa = clamp(a - 1 + (ground - footLow[a - 1]) / v, a - 1, a); }
      }
      if (b < hi) {
        const y0 = footLow[b], y1 = footLow[b + 1]; if (y1 !== y0) fb = b + clamp((y0 - (ground - thrPx)) / (y0 - y1), 0, 1);
        if (b + 2 <= hi) { const v = footLow[b + 1] - footLow[b + 2]; if (v > 0.3 * thrPx) fb = clamp(b + 1 - (ground - footLow[b + 1]) / v, b, b + 1); }
      }
      const ev = { side, ic_i: a, to_i: b, ic: tAt(t, fa), to: tAt(t, fb), plateau_ic: tAt(t, fa), plateau_to: tAt(t, fb), method: "plateau", lowest_i: Math.round((a + b) / 2), truncated: a <= lo || b >= hi };
      if (view === "side") {
        // Zeni (2008): IC = heel most anterior relative to the pelvis, TO = toe most posterior. Searched inside this cycle.
        let zi = lo; for (let i = lo; i <= Math.min(hi, i0 + 1); i++) if (relHeel[i] > relHeel[zi]) zi = i;
        let zt = i0; for (let i = Math.max(lo, i0 - 1); i <= hi; i++) if (relToe[i] < relToe[zt]) zt = i;
        ev.zeni_ic = tAt(t, refinePeak(relHeel, zi)); ev.zeni_to = tAt(t, refinePeak(relToe.map((v) => -v), zt));
        ev.zeni_ic_i = zi; ev.zeni_to_i = zt;
        // Fusion. Both IC estimators are biased EARLY (heel-max precedes contact during paw-back; the height
        // threshold is crossed while the foot is still descending), so take the later of the two when they agree
        // (<= 80 ms apart); otherwise the plateau is untrustworthy (camera shake / perspective) -> Zeni + 15 ms.
        // TO: Zeni toe-min and the plateau end agree within a frame in clean footage -> mean; else Zeni.
        const dIC = ev.plateau_ic - ev.zeni_ic;
        if (Math.abs(dIC) <= 0.035) ev.ic = (ev.plateau_ic + ev.zeni_ic) / 2; // noise regime -> average
        else if (Math.abs(dIC) <= 0.08) { ev.ic = Math.max(ev.plateau_ic, ev.zeni_ic); ev.method = dIC >= 0 ? "plateau" : "zeni"; } // paw-back regime
        else { ev.ic = ev.zeni_ic + 0.015; ev.method = "zeni"; }
        const dTO = ev.plateau_to - ev.zeni_to;
        if (Math.abs(dTO) <= 0.04) ev.to = (ev.plateau_to + ev.zeni_to) / 2; else { ev.to = ev.zeni_to; ev.method = "zeni"; }
      }
      events.push(ev);
    }
    // drop truncated cycles at the clip edges (unknown IC or TO)
    return events.filter((e) => !e.truncated && e.to > e.ic);
  }

  // ------------------------------------------------------------ main engine
  /**
   * analyzeFrames(frames, opts) -> result
   * opts: {W, H, fps, heightCm, weightKg, slowmo: 'auto'|number, view: 'auto'|'side'|'front'|'back', refCadence?, speed? (m/s)}
   */
  function analyzeFrames(frames, opts) {
    const W = opts.W, H = opts.H, heightCm = opts.heightCm || 171, weightKg = opts.weightKg || 61.5;
    const warnings = [];
    if (!frames || frames.length < 10) throw new Error("影片太短或偵測不到人體姿勢");
    const { P, quality, okFrame } = preprocess(frames, W, H);
    const n = frames.length; const tv = frames.map((f) => f.t);
    if (quality < 0.6) warnings.push(`僅 ${Math.round(quality * 100)}% 的影格偵測到完整下肢，結果可信度較低（請確認全身入鏡、光線充足）`);
    const vd = detectView(P, n);
    let view = opts.view && opts.view !== "auto" ? opts.view : vd.view;
    if (view === "oblique") { warnings.push("鏡頭角度介於側面與正面之間（斜角），角度類指標會有偏差；建議正側面 90° 拍攝"); view = vd.ratio < 0.5 ? "side" : (vd.noseVis > 0.6 ? "front" : "back"); }
    const dir = vd.dir; const scale = bodyScale(P, n, heightCm, view); const pxPerCm = scale.pxPerCm;
    // --- stride period via autocorrelation of ankle-height difference (robust to camera motion)
    const fpsEff = (n - 1) / Math.max(1e-6, tv[n - 1] - tv[0]);
    const diffSig = []; for (let i = 0; i < n; i++) diffSig.push(P[27].y[i] - P[28].y[i]);
    const maxLag = Math.min(Math.floor(n / 2.2), Math.round(fpsEff * 9));
    const dp = dominantPeriod(diffSig, Math.max(3, Math.round(fpsEff * 0.35)), maxLag);
    if (!dp) throw new Error("偵測不到週期性的跑步動作（請確認影片中有連續跑步，且至少 3 個完整步幅）");
    const strideFrames = dp.period; // frames per stride (2 steps) in video time
    const cadRawVideo = 120 / (strideFrames / fpsEff); // spm in video time
    // --- slow-motion factor
    let slow = 1;
    if (opts.slowmo && opts.slowmo !== "auto") slow = Number(opts.slowmo) || 1;
    else {
      const cands = [1, 2, 4, 8].map((k) => ({ k, cad: cadRawVideo * k })).filter((c) => c.cad >= 130 && c.cad <= 215);
      const ref = opts.refCadence || 172;
      slow = cands.length ? cands.reduce((p, q) => (Math.abs(q.cad - ref) < Math.abs(p.cad - ref) ? q : p)).k : 1;
      if (!cands.length) warnings.push(`偵測到的步頻 ${Math.round(cadRawVideo)} spm 不在合理範圍，慢動作倍率無法自動判定（已視為 1×）`);
    }
    const t = tv.map((x) => (x - tv[0]) / slow); // real seconds
    const fpsReal = fpsEff * slow; const dtMs = 1000 / fpsReal;
    const cadence = cadRawVideo * slow;
    const stridePeriodS = strideFrames / fpsEff / slow;
    // --- events
    const evOpts = { stridePeriod: strideFrames, pxPerCm, dir, view };
    const evL = detectEvents(P, t, "L", evOpts), evR = detectEvents(P, t, "R", evOpts);
    const all = evL.concat(evR).sort((a, b) => a.ic - b.ic);
    // pair consecutive alternating events
    const steps = [];
    for (let i = 0; i < all.length - 1; i++) {
      const a = all[i], b = all[i + 1];
      if (a.side === b.side) continue;
      const stepT = b.ic - a.ic; if (stepT < 0.2 || stepT > 0.7) continue;
      steps.push({ from: a.side, stepT, gct: a.to - a.ic, flight: b.ic - a.to, ic: a.ic, to: a.to, ev: a });
    }
    const gctOf = (side) => all.filter((e) => e.side === side).map((e) => (e.to - e.ic) * 1000).filter((v) => v > 80 && v < 800);
    const gctL = gctOf("L"), gctR = gctOf("R");
    const stepTimes = steps.map((s) => s.stepT);
    const cadEvents = stepTimes.length >= 3 ? 60 / median(stepTimes) : null;
    const flights = steps.map((s) => s.flight * 1000).filter((v) => v > -150 && v < 300);
    const gct = median(gctL.concat(gctR));
    const gctPlateau = median(all.map((e) => (e.plateau_to - e.plateau_ic) * 1000).filter((v) => v > 80 && v < 500));
    const gctZeni = view === "side" ? median(all.map((e) => (e.zeni_to - e.zeni_ic) * 1000).filter((v) => v > 80 && v < 500)) : null;
    const nZeni = all.filter((e) => e.method === "zeni").length;
    if (view === "side" && isNum(gctPlateau) && isNum(gctZeni) && Math.abs(gctPlateau - gctZeni) > 45) warnings.push(`兩種著地偵測法的觸地時間差 ${Math.round(Math.abs(gctPlateau - gctZeni))} ms（足底高度法 ${Math.round(gctPlateau)} / 骨盆相對法 ${Math.round(gctZeni)}），可能有鏡頭晃動或透視變形，GCT 僅供參考`);
    const flight = flights.length ? median(flights) : null;
    const walking = flight !== null && flight < 0 && steps.length >= 3;
    if (walking) warnings.push("偵測到雙腳同時著地（無騰空期）— 這段看起來是走路或極慢跑，跑姿指標不適用");
    if (all.length < 4) warnings.push(`只偵測到 ${all.length} 次著地，請拍攝至少 8–10 步（10–20 秒）`);
    if (fpsReal < 50) warnings.push(`影片 ${Math.round(fpsReal)} fps：觸地/騰空時間的解析度約 ±${Math.round(dtMs)} ms；建議以 60 fps 或慢動作 120/240 fps 拍攝`);
    if (cadEvents && Math.abs(cadEvents - cadence) / cadence > 0.12) warnings.push(`事件法步頻 (${Math.round(cadEvents)}) 與週期法步頻 (${Math.round(cadence)}) 差異較大，著地偵測可能受遮蔽或鏡頭晃動影響`);
    // --- vertical oscillation (pelvis centre, high-passed over one stride, per step peak-to-peak)
    const pelvY = []; for (let i = 0; i < n; i++) pelvY.push((P[23].y[i] + P[24].y[i]) / 2);
    const hp = pelvY.map((v, i) => v - movingAverage(pelvY, strideFrames)[i]);
    const voList = [];
    const idxAt = (tt) => { let i = 0; while (i < n - 1 && t[i + 1] <= tt) i++; return i; };
    for (const s of steps) { const i0 = idxAt(s.ic), i1 = idxAt(s.ic + s.stepT); if (i1 - i0 < 2) continue; let mx = -Infinity, mn = Infinity; for (let i = i0; i <= i1; i++) { mx = Math.max(mx, hp[i]); mn = Math.min(mn, hp[i]); } voList.push((mx - mn) / pxPerCm); }
    if (!voList.length) { // fallback: whole-clip amplitude of the step-frequency component
      const half = Math.max(2, Math.round(strideFrames / 2)); for (let i = 0; i + half < n; i += half) { let mx = -Infinity, mn = Infinity; for (let j = i; j < i + half; j++) { mx = Math.max(mx, hp[j]); mn = Math.min(mn, hp[j]); } voList.push((mx - mn) / pxPerCm); }
    }
    const vo = median(voList);
    // --- lateral sway (front/back): pelvis x high-passed
    const pelvX = []; for (let i = 0; i < n; i++) pelvX.push((P[23].x[i] + P[24].x[i]) / 2);
    const hpx = pelvX.map((v, i) => v - movingAverage(pelvX, strideFrames)[i]);
    const sway = (pctl(hpx, 0.95) - pctl(hpx, 0.05)) / pxPerCm;

    const res = { view, dir, nearSide: vd.nearSide, viewRatio: vd.ratio, quality: r1(quality * 100), fps: r1(fpsReal), fpsVideo: r1(fpsEff), slowmo: slow, pxPerCm: r1(pxPerCm * 100) / 100,
      nFrames: n, duration: r1(t[n - 1]), nContacts: all.length, nSteps: steps.length, warnings, walking,
      cadence: r1(cadence), cadenceEvents: r1(cadEvents), autocorr: r1(dp.r * 100) / 100,
      gct: r0(gct), gctL: r0(median(gctL)), gctR: r0(median(gctR)), gctSd: r0(sd(gctL.concat(gctR))), gctTol: r0(dtMs * 0.7), gctPlateau: r0(gctPlateau), gctZeni: r0(gctZeni), eventMethod: nZeni > all.length / 2 ? "zeni" : "plateau",
      flight: r0(flight), stepTime: r0(median(stepTimes) * 1000), strideTime: r0(stridePeriodS * 1000),
      vo: r1(vo), voSd: r1(sd(voList)), sway: r1(sway), events: all.map((e) => ({ side: e.side, ic: r1(e.ic * 1000) / 1000, to: r1(e.to * 1000) / 1000, gct: r0((e.to - e.ic) * 1000), icVideoT: r1((e.ic * slow + tv[0]) * 1000) / 1000, toVideoT: r1((e.to * slow + tv[0]) * 1000) / 1000, lowVideoT: r1(tv[e.lowest_i] * 1000) / 1000 })),
      steps: steps.map((s) => ({ from: s.from, stepT: r0(s.stepT * 1000), gct: r0(s.gct * 1000), flight: r0(s.flight * 1000) })), voList: voList.map(r1) };
    if (gct && res.stepTime) res.dutyFactor = r1(gct / (2 * res.stepTime) * 100) / 100; // contact / stride time
    if (gctL.length && gctR.length) res.gctAsym = r1(Math.abs(median(gctL) - median(gctR)) / mean([median(gctL), median(gctR)]) * 100);
    const stL = steps.filter((s) => s.from === "L").map((s) => s.stepT), stR = steps.filter((s) => s.from === "R").map((s) => s.stepT);
    if (stL.length && stR.length) res.stepAsym = r1(Math.abs(median(stL) - median(stR)) / mean([median(stL), median(stR)]) * 100);

    if (view === "side") Object.assign(res, sagittalMetrics(P, t, n, all, dir, pxPerCm, vd.nearSide, strideFrames));
    else Object.assign(res, frontalMetrics(P, t, n, all, view, pxPerCm, strideFrames));
    // --- arm swing (elbow angle) in every view
    const elb = { L: [], R: [] };
    for (let i = 0; i < n; i++) for (const s of ["L", "R"]) { const S = SIDES[s]; if (P[S.wr].v[i] > 0.5 && P[S.el].v[i] > 0.5) elb[s].push(angleAt(pt(P, S.sh, i), pt(P, S.el, i), pt(P, S.wr, i))); }
    res.elbowL = r0(median(elb.L)); res.elbowR = r0(median(elb.R)); res.elbow = r0(median(elb.L.concat(elb.R)));
    // --- speed-dependent metrics
    if (isNum(opts.speed) && opts.speed > 1) Object.assign(res, speedMetrics(res, opts.speed, heightCm, weightKg));
    res.heightCm = heightCm; res.weightKg = weightKg;
    return res;
  }

  // ------------------------------------------------------- sagittal metrics
  function sagittalMetrics(P, t, n, events, dir, pxPerCm, nearSide, strideFrames) {
    const out = { perSide: {} };
    const idxAt = (tt) => { let best = 0, bd = Infinity; for (let i = 0; i < n; i++) { const d = Math.abs(t[i] - tt); if (d < bd) { bd = d; best = i; } } return best; };
    const idxFloor = (tt) => { let i = 0; while (i < n - 1 && t[i + 1] <= tt + 1e-6) i++; return i; };
    const trunkLean = [], headFwd = [];
    for (let i = 0; i < n; i++) {
      const sh = mid(P, 11, 12, i), hip = mid(P, 23, 24, i);
      trunkLean.push(Math.atan2(dir * (sh.x - hip.x), hip.y - sh.y) * DEG);
      const ear = mid(P, 7, 8, i); headFwd.push(Math.atan2(dir * (ear.x - sh.x), sh.y - ear.y) * DEG);
    }
    out.trunkLean = r1(median(trunkLean)); out.trunkLeanSd = r1(sd(trunkLean)); out.headFwd = r1(median(headFwd) - median(trunkLean));
    for (const side of ["L", "R"]) {
      const S = SIDES[side]; const evs = events.filter((e) => e.side === side);
      const m = { shankIC: [], kneeIC: [], footIC: [], footAhead: [], kneeStance: [], kneeSwing: [], thighTO: [], hipExtTO: [], kneeCurve: [], armAmp: [] };
      const kneeFlex = (i) => 180 - angleAt(pt(P, S.hip, i), pt(P, S.knee, i), pt(P, S.ank, i));
      for (let k = 0; k < evs.length; k++) {
        const e = evs[k]; const ic = idxAt(e.ic), to = idxAt(e.to), icf = idxFloor(e.ic), icc = Math.min(n - 1, icf + 1);
        // landing posture = mean of the frames bracketing the contact instant (robust to ±1 frame timing error)
        const br = (fn) => mean([fn(icf), fn(icc)]);
        m.shankIC.push(br((i) => Math.atan2(dir * (P[S.ank].x[i] - P[S.knee].x[i]), P[S.ank].y[i] - P[S.knee].y[i]) * DEG));
        m.kneeIC.push(br(kneeFlex));
        m.footIC.push(br((i) => Math.atan2(-(P[S.toe].y[i] - P[S.heel].y[i]), dir * (P[S.toe].x[i] - P[S.heel].x[i])) * DEG));
        m.footAhead.push(br((i) => dir * (P[S.ank].x[i] - (P[23].x[i] + P[24].x[i]) / 2) / pxPerCm));
        const hip = pt(P, S.hip, icf);
        let ks = -Infinity; for (let i = ic; i <= to; i++) ks = Math.max(ks, kneeFlex(i)); m.kneeStance.push(ks);
        const toK = pt(P, S.knee, to), toH = pt(P, S.hip, to), toS = pt(P, S.sh, to);
        m.thighTO.push(Math.atan2(dir * (toK.x - toH.x), toK.y - toH.y) * DEG); // negative = knee behind hip
        m.hipExtTO.push(angleAt(toS, toH, toK) - 180); // >0 = thigh behind trunk line
        const next = evs[k + 1]; const swEnd = next ? idxAt(next.ic) : Math.min(n - 1, to + Math.round(strideFrames * 0.7));
        let kw = -Infinity; for (let i = to; i <= swEnd; i++) kw = Math.max(kw, kneeFlex(i)); if (swEnd > to + 2) m.kneeSwing.push(kw);
        if (next && k === Math.floor(evs.length / 2)) { const i1 = idxAt(next.ic); const curve = []; for (let i = ic; i <= i1; i++) curve.push({ p: r0((i - ic) / Math.max(1, i1 - ic) * 100), k: r0(kneeFlex(i)) }); m.kneeCurve = curve; m.stancePct = r0((to - ic) / Math.max(1, i1 - ic) * 100); }
        void hip;
      }
      const wrRel = []; for (let i = 0; i < n; i++) if (P[S.wr].v[i] > 0.5) wrRel.push(dir * (P[S.wr].x[i] - P[S.sh].x[i]) / pxPerCm);
      out.perSide[side] = { n: evs.length, reliable: side === nearSide, shankIC: r1(median(m.shankIC)), kneeIC: r1(median(m.kneeIC)), footIC: r1(median(m.footIC)), footAhead: r1(median(m.footAhead)),
        kneeStance: r1(median(m.kneeStance)), kneeSwing: r1(median(m.kneeSwing)), thighTO: r1(median(m.thighTO)), hipExtTO: r1(median(m.hipExtTO)), kneeCurve: m.kneeCurve, stancePct: m.stancePct,
        armAmp: wrRel.length ? r1(pctl(wrRel, 0.95) - pctl(wrRel, 0.05)) : null, armFront: wrRel.length ? r1(pctl(wrRel, 0.95)) : null, armBack: wrRel.length ? r1(pctl(wrRel, 0.05)) : null };
    }
    // headline values: prefer the near (camera-side) leg, fall back to the mean of both
    const near = out.perSide[nearSide], far = out.perSide[nearSide === "L" ? "R" : "L"];
    const pick = (k) => (isNum(near[k]) ? near[k] : far[k]);
    const both = (k) => mean([near[k], far[k]]);
    out.shankIC = pick("shankIC"); out.kneeIC = pick("kneeIC"); out.footIC = pick("footIC"); out.footAhead = pick("footAhead"); out.kneeStance = pick("kneeStance"); out.kneeSwing = pick("kneeSwing"); out.thighTO = pick("thighTO"); out.hipExtTO = pick("hipExtTO");
    out.hipExt = isNum(out.thighTO) ? r1(-out.thighTO) : null; // thigh angle behind vertical at toe-off
    out.armAmp = r1(both("armAmp"));
    // foot strike angle bands after Altman & Davis (2012): rearfoot > 8°, midfoot −1.6…8°, forefoot < −1.6°
    out.footStrike = !isNum(out.footIC) ? null : out.footIC > 8 ? "heel" : out.footIC < -1.6 ? "forefoot" : "midfoot";
    out.kneeSwingAsym = isNum(near.kneeSwing) && isNum(far.kneeSwing) ? r1(Math.abs(near.kneeSwing - far.kneeSwing)) : null;
    return out;
  }

  // -------------------------------------------------------- frontal metrics
  function frontalMetrics(P, t, n, events, view, pxPerCm, strideFrames) {
    const out = { perSide: {} };
    const idxAt = (tt) => { let best = 0, bd = Infinity; for (let i = 0; i < n; i++) { const d = Math.abs(t[i] - tt); if (d < bd) { bd = d; best = i; } } return best; };
    // image-x sign of the athlete's LEFT side: front view -> left side appears on image right (+1)
    const leftSign = view === "front" ? 1 : -1;
    const cross = { L: 0, R: 0, n: 0 };
    for (let i = 0; i < n; i++) {
      const px = (P[23].x[i] + P[24].x[i]) / 2; cross.n++;
      // wrist across the body midline by > 3 cm
      if (P[15].v[i] > 0.5 && leftSign * (P[15].x[i] - px) < -3 * pxPerCm) cross.L++;
      if (P[16].v[i] > 0.5 && -leftSign * (P[16].x[i] - px) < -3 * pxPerCm) cross.R++;
    }
    out.armCrossL = r0(cross.L / cross.n * 100); out.armCrossR = r0(cross.R / cross.n * 100); out.armCross = r0((cross.L + cross.R) / (2 * cross.n) * 100);
    const shoulderTilt = [], hipTiltAll = [];
    for (let i = 0; i < n; i++) { shoulderTilt.push(Math.atan2(P[11].y[i] - P[12].y[i], Math.abs(P[11].x[i] - P[12].x[i])) * DEG); hipTiltAll.push(Math.atan2(P[23].y[i] - P[24].y[i], Math.abs(P[23].x[i] - P[24].x[i])) * DEG); }
    out.shoulderTilt = r1(median(shoulderTilt)); // + = left shoulder lower (image)
    for (const side of ["L", "R"]) {
      const S = SIDES[side], O = SIDES[side === "L" ? "R" : "L"]; const evs = events.filter((e) => e.side === side);
      const drop = [], fppa = [], lateral = [], kneeSep = [];
      for (const e of evs) {
        const ic = idxAt(e.ic), to = idxAt(e.to); const ms = Math.round((ic + to) / 2);
        // contralateral pelvic drop at midstance: swing-side hip lower (larger y) => positive
        drop.push(Math.atan2(P[O.hip].y[ms] - P[S.hip].y[ms], Math.abs(P[O.hip].x[ms] - P[S.hip].x[ms])) * DEG);
        // frontal-plane projection angle at the knee; medial (valgus) positive
        const hip = pt(P, S.hip, ms), knee = pt(P, S.knee, ms), ank = pt(P, S.ank, ms);
        const a = 180 - angleAt(hip, knee, ank);
        // signed distance of the knee from the hip-ankle line; medial = towards the other hip
        const vx = ank.x - hip.x, vy = ank.y - hip.y; const L = Math.hypot(vx, vy) || 1; const d = ((knee.x - hip.x) * vy - (knee.y - hip.y) * vx) / L;
        const toOther = Math.sign(P[O.hip].x[ms] - hip.x); const medial = Math.sign(-d * vy) === toOther ? 1 : -1; // orientation of perpendicular
        fppa.push(medial * a);
        // lateral position of the landing foot relative to the pelvis centre: + = outside (normal), - = crossed midline
        const px = (P[23].x[ic] + P[24].x[ic]) / 2; const sideSign = side === "L" ? leftSign : -leftSign;
        lateral.push(sideSign * (P[S.ank].x[ic] - px) / pxPerCm);
        kneeSep.push(Math.abs(P[25].x[ms] - P[26].x[ms]) / pxPerCm);
      }
      out.perSide[side] = { n: evs.length, pelvicDrop: r1(median(drop)), fppa: r1(median(fppa)), footLateral: r1(median(lateral)), kneeSep: r1(median(kneeSep)) };
    }
    const L = out.perSide.L, R = out.perSide.R;
    // camera roll cancels in the mean of left/right drops
    out.pelvicDrop = isNum(L.pelvicDrop) && isNum(R.pelvicDrop) ? r1((L.pelvicDrop + R.pelvicDrop) / 2) : (L.pelvicDrop ?? R.pelvicDrop);
    out.pelvicDropAsym = isNum(L.pelvicDrop) && isNum(R.pelvicDrop) ? r1((L.pelvicDrop - R.pelvicDrop) / 2) : null;
    out.fppa = r1(Math.max(L.fppa ?? -99, R.fppa ?? -99)); if (out.fppa === -99) out.fppa = null;
    out.footLateral = r1(Math.min(L.footLateral ?? 99, R.footLateral ?? 99)); if (out.footLateral === 99) out.footLateral = null;
    out.stepWidth = isNum(L.footLateral) && isNum(R.footLateral) ? r1(L.footLateral + R.footLateral) : null;
    return out;
  }

  // ------------------------------------------------- speed-dependent metrics
  function speedMetrics(res, v, heightCm, weightKg) {
    const o = { speed: r1(v * 100) / 100, pace_s: r0(1000 / v) };
    if (res.cadence) o.stepLen = r1(v / (res.cadence / 60) * 100) / 100; // m per step
    if (isNum(res.vo) && o.stepLen) o.vr = r1(res.vo / (o.stepLen * 100) * 100); // %
    if (o.stepLen) o.stepLenPctHeight = r0(o.stepLen * 100 / heightCm * 100);
    if (res.gct && isNum(res.flight) && res.flight > 0) {
      const tc = res.gct / 1000, tf = res.flight / 1000, m = weightKg;
      const Fmax = m * G * (Math.PI / 2) * (tf / tc + 1);
      const dz = Fmax * tc * tc / (m * Math.PI * Math.PI) - G * tc * tc / 8;
      const Lleg = SEG.leg * heightCm / 100;
      const dL = Lleg - Math.sqrt(Math.max(0, Lleg * Lleg - Math.pow(v * tc / 2, 2))) + dz;
      o.Fmax_bw = r1(Fmax / (m * G) * 100) / 100; o.comDrop = r1(dz * 100); o.kvert = r1(Fmax / dz / 1000); o.kleg = r1(Fmax / dL / 1000);
    }
    if (isNum(res.vo) && res.cadence) o.vertLiftPerMin = r1(res.vo / 100 * res.cadence);
    return o;
  }

  global.FORM = Object.assign(global.FORM || {}, { analyzeFrames, preprocess, detectView, bodyScale, detectEvents, speedMetrics, util: { mean, median, pctl, sd, dominantPeriod, localMaxima, movingAverage }, LM, SIDES, CONNECTIONS, SEG });
})(typeof window !== "undefined" ? window : globalThis);


/* ============================================================================
 * Part 2 — scoring, coaching expert system, COROS matching, video metadata, UI
 * ========================================================================== */
(function (global) {
  "use strict";
  const F = global.FORM;
  const isNum = (v) => typeof v === "number" && Number.isFinite(v);
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const r1 = (v) => (isNum(v) ? Math.round(v * 10) / 10 : null);
  const r0 = (v) => (isNum(v) ? Math.round(v) : null);
  const h = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const paceStr = (s) => (!isNum(s) || s <= 0 ? "-" : `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`);
  /** piecewise-linear score through sorted [x, score] points */
  function pl(v, pts) { if (!isNum(v)) return null; if (v <= pts[0][0]) return pts[0][1]; for (let i = 1; i < pts.length; i++) { if (v <= pts[i][0]) { const [x0, s0] = pts[i - 1], [x1, s1] = pts[i]; return s0 + (s1 - s0) * (v - x0) / (x1 - x0); } } return pts[pts.length - 1][1]; }

  // ------------------------------------------------------------- targets
  // pace-dependent benchmarks (pace in s/km; default = easy pace when unknown)
  const cadT = (p) => clamp(184 - 0.06 * (p - 240), 166, 184);
  const gctGood = (p) => 160 + 0.25 * p; // ms; ~219 @3:55, 235 @5:00, 250 @6:00
  const kfT = (p) => clamp(125 - 0.2 * (p - 200), 85, 125); // peak swing knee flexion

  // ------------------------------------------------------------- metrics
  // key, label, unit, views, weight, value(res), score(v, ctx), target text, fmt
  const METRICS = [
    { key: "cadence", label: "步頻", unit: "spm", views: ["side", "front", "back"], w: 14, val: (r) => r.cadence,
      score: (v, c) => { const T = cadT(c.pace); if (v > T + 14) return 85; return pl(v, [[T - 22, 25], [T - 14, 50], [T - 6, 80], [T - 1, 100]]); },
      target: (c) => `≥ ${Math.round(cadT(c.pace) - 2)} spm（此配速）` },
    { key: "gct", label: "觸地時間", unit: "ms", views: ["side", "front", "back"], w: 14, val: (r) => r.gct,
      score: (v, c) => { const g = gctGood(c.pace); return pl(v, [[g - 25, 100], [g, 90], [g + 20, 70], [g + 45, 45], [g + 70, 25]]); },
      target: (c) => `≤ ${Math.round(gctGood(c.pace))} ms（此配速）`, fmt: (v, r) => `${v}${r.gctTol ? ` ±${r.gctTol}` : ""}` },
    { key: "dutyFactor", label: "觸地佔比 (duty factor)", unit: "", views: ["side", "front", "back"], w: 4, val: (r) => r.dutyFactor,
      score: (v) => pl(v, [[0.33, 100], [0.36, 85], [0.40, 65], [0.45, 40], [0.5, 25]]), target: () => "≤ 0.36（觸地 / 完整步幅週期）" },
    { key: "vo", label: "垂直振幅", unit: "cm", views: ["side", "front", "back"], w: 10, val: (r) => r.vo,
      score: (v) => pl(v, [[6.5, 100], [8, 90], [9.5, 70], [11, 45], [13, 25]]), target: () => "6–8 cm" },
    { key: "vr", label: "垂直比 (振幅 / 步長)", unit: "%", views: ["side", "front", "back"], w: 8, val: (r) => r.vr,
      score: (v) => pl(v, [[6, 100], [7.4, 88], [8.6, 70], [10, 45], [12, 25]]), target: () => "< 7.5 %" },
    { key: "shankIC", label: "著地小腿角（過度跨步指標）", unit: "°", views: ["side"], w: 14, val: (r) => r.shankIC,
      score: (v) => (v < -3 ? 95 : pl(v, [[3, 100], [6, 90], [10, 65], [14, 40], [20, 20]])), target: () => "0–6°（腳踝在膝蓋正下方附近著地）" },
    { key: "footAhead", label: "著地點在骨盆前方", unit: "cm", views: ["side"], w: 0, val: (r) => r.footAhead, score: () => null, target: () => "參考值（配速越快越大）" },
    { key: "kneeIC", label: "著地膝彎曲", unit: "°", views: ["side"], w: 4, val: (r) => r.kneeIC,
      score: (v) => pl(v, [[4, 50], [8, 75], [12, 100], [25, 100], [32, 80], [42, 60]]), target: () => "12–25°（不要鎖直）" },
    { key: "footStrike", label: "著地方式", unit: "", views: ["side"], w: 4, val: (r) => r.footStrike,
      score: (v, c, r) => (v === "midfoot" ? 100 : v === "forefoot" ? 95 : v === "heel" ? (isNum(r.shankIC) && r.shankIC <= 6 ? 85 : 55) : null),
      target: () => "中足／前足，或小腿角小的近端腳跟著地", fmt: (v, r) => ({ heel: "後足（腳跟）", midfoot: "中足", forefoot: "前足" }[v] || "-") + (isNum(r.footIC) ? ` · 足角 ${r.footIC}°` : "") },
    { key: "trunkLean", label: "軀幹前傾", unit: "°", views: ["side"], w: 8, val: (r) => r.trunkLean,
      score: (v) => pl(v, [[-2, 40], [0, 55], [3, 80], [5, 100], [10, 100], [13, 80], [16, 55], [20, 35]]), target: () => "5–10°（由腳踝前傾，非彎腰）" },
    { key: "hipExt", label: "離地髖伸展（大腿後擺）", unit: "°", views: ["side"], w: 8, val: (r) => r.hipExt,
      score: (v) => pl(v, [[0, 30], [6, 55], [10, 75], [15, 92], [20, 100]]), target: () => "≥ 15°（大腿在垂直線後方）" },
    { key: "kneeSwing", label: "後擺膝折疊（腳跟回收）", unit: "°", views: ["side"], w: 5, val: (r) => r.kneeSwing,
      score: (v, c) => { const T = kfT(c.pace); return pl(v, [[T - 35, 40], [T - 20, 65], [T - 8, 85], [T, 100]]); }, target: (c) => `≥ ${Math.round(kfT(c.pace) - 5)}°（此配速）` },
    { key: "kneeStance", label: "支撐期最大膝彎曲", unit: "°", views: ["side"], w: 2, val: (r) => r.kneeStance,
      score: (v) => pl(v, [[25, 70], [35, 100], [50, 100], [60, 70]]), target: () => "35–50°（吸震但不塌陷）" },
    { key: "elbow", label: "手肘角度", unit: "°", views: ["side", "front", "back"], w: 4, val: (r) => r.elbow,
      score: (v) => pl(v, [[55, 55], [70, 85], [80, 100], [105, 100], [115, 85], [130, 65], [150, 45]]), target: () => "80–105°" },
    { key: "armAmp", label: "手臂擺幅（腕相對肩）", unit: "cm", views: ["side"], w: 2, val: (r) => r.armAmp,
      score: (v) => pl(v, [[12, 55], [20, 85], [28, 100], [50, 100], [62, 75]]), target: () => "28–50 cm" },
    { key: "headFwd", label: "頭部前伸（相對軀幹）", unit: "°", views: ["side"], w: 2, val: (r) => r.headFwd,
      score: (v) => pl(Math.abs(v), [[8, 100], [15, 70], [25, 45]]), target: () => "±8°" },
    { key: "trunkLeanSd", label: "軀幹穩定度（前傾角波動）", unit: "°", views: ["side"], w: 2, val: (r) => r.trunkLeanSd,
      score: (v) => pl(v, [[2.5, 100], [4, 80], [6, 55], [9, 35]]), target: () => "< 3°" },
    { key: "gctAsym", label: "左右觸地對稱", unit: "%", views: ["side", "front", "back"], w: 6, val: (r) => r.gctAsym,
      score: (v) => pl(v, [[3, 100], [5, 88], [8, 70], [12, 45], [18, 25]]), target: () => "< 5 %", fmt: (v, r) => `${v} (L ${r.gctL ?? "-"} / R ${r.gctR ?? "-"} ms)` },
    { key: "pelvicDrop", label: "骨盆下沉（對側）", unit: "°", views: ["front", "back"], w: 16, val: (r) => r.pelvicDrop,
      score: (v) => pl(v, [[4, 100], [6, 85], [8, 65], [11, 40], [15, 20]]), target: () => "< 5°", fmt: (v, r) => `${v} (L支撐 ${r.perSide?.L?.pelvicDrop ?? "-"} / R支撐 ${r.perSide?.R?.pelvicDrop ?? "-"})` },
    { key: "fppa", label: "膝內扣（額狀面投影角）", unit: "°", views: ["front", "back"], w: 14, val: (r) => r.fppa,
      score: (v) => pl(v, [[6, 100], [10, 80], [14, 55], [20, 30]]), target: () => "< 8°", fmt: (v, r) => `${v} (L ${r.perSide?.L?.fppa ?? "-"} / R ${r.perSide?.R?.fppa ?? "-"})` },
    { key: "footLateral", label: "著地橫向位置（交叉步）", unit: "cm", views: ["front", "back"], w: 10, val: (r) => r.footLateral,
      score: (v) => pl(v, [[-4, 30], [-1, 55], [1.5, 80], [3, 100]]), target: () => "> 2 cm（腳不過身體中線）", fmt: (v, r) => `${v} (L ${r.perSide?.L?.footLateral ?? "-"} / R ${r.perSide?.R?.footLateral ?? "-"}；步寬 ${r.stepWidth ?? "-"})` },
    { key: "armCross", label: "手臂越過中線", unit: "%", views: ["front", "back"], w: 8, val: (r) => r.armCross,
      score: (v) => pl(v, [[8, 100], [20, 75], [40, 45], [60, 25]]), target: () => "< 10 % 的時間", fmt: (v, r) => `${v} (L ${r.armCrossL ?? "-"} / R ${r.armCrossR ?? "-"})` },
    { key: "sway", label: "骨盆側向擺動", unit: "cm", views: ["front", "back"], w: 6, val: (r) => r.sway,
      score: (v) => pl(v, [[4, 100], [6, 85], [9, 60], [13, 35]]), target: () => "< 5 cm" },
    { key: "shoulderTilt", label: "肩膀水平", unit: "°", views: ["front", "back"], w: 3, val: (r) => r.shoulderTilt,
      score: (v) => pl(Math.abs(v), [[3, 100], [6, 75], [10, 50]]), target: () => "±3°" },
  ];

  function scoreResult(res, ctx) {
    const pace = ctx && isNum(ctx.pace_s) ? ctx.pace_s : (isNum(res.pace_s) ? res.pace_s : 330);
    const c = { pace, known: !!(ctx && isNum(ctx.pace_s)) || isNum(res.pace_s) };
    const rows = []; let sw = 0, ss = 0;
    for (const m of METRICS) {
      if (!m.views.includes(res.view)) continue;
      const v = m.val(res); if (v === null || v === undefined) continue;
      const s = m.score(v, c, res);
      const row = { key: m.key, label: m.label, unit: m.unit, value: v, score: isNum(s) ? Math.round(s) : null, w: m.w, target: m.target(c), text: m.fmt ? m.fmt(v, res) : (isNum(v) ? `${v}` : String(v)) };
      rows.push(row);
      if (isNum(s) && m.w > 0) { sw += m.w; ss += m.w * s; }
    }
    const total = sw ? Math.round(ss / sw) : null;
    const grade = total === null ? "-" : total >= 88 ? "A" : total >= 82 ? "A−" : total >= 76 ? "B+" : total >= 70 ? "B" : total >= 60 ? "C" : "D";
    return { total, grade, rows, paceUsed: pace, paceKnown: c.known };
  }

  // -------------------------------------------------------- expert system
  const KNOWLEDGE = {
    cadence: { title: "步頻偏低", why: "同一速度下步頻偏低 = 每一步跨得更長：著地點前移、制動衝量增加、觸地時間與垂直振幅跟著變大。Heiderscheit (2011) 顯示步頻提高 5–10% 可同時降低垂直位移、制動力與髖膝負荷，且幾乎不增加能量消耗；半馬後段疲勞時，步頻通常先掉，這會直接吃掉跑步經濟性。",
      cue: "「腳落在身體下方、快而輕」；用 COROS 的步頻提示或節拍器，先加 3–5%，不要一次加 10%。",
      drills: ["節拍器輕鬆跑：2×10 分鐘 @ 目前步頻 +5%，每週 2 次（放在 E 日）", "大步跑 6×20 秒：強調快速落地回收，而不是拉大步", "跑步機 1% 坡度 3×3 分鐘，用螢幕步頻回饋鎖定目標步頻"], plan: "先在輕鬆日練，閾值課與長跑維持自然步頻；2–3 週後再檢查自選步頻是否上移。" },
    gct: { title: "觸地時間偏長", why: "觸地越久，制動期越長、肌腱（阿基里斯腱／足弓）回彈的彈性能量越少，必須用更多肌肉收縮來推進。多篇研究 (Santos-Concejero 2014, Moore 2016) 指出同速度下 GCT 越短、跑步經濟性越好；COROS 的觸地時間與影片估計都可以追蹤這個趨勢。",
      cue: "「踩到熱鐵、腳輕快離地」；想像地面是彈簧，不是用腳推地。",
      drills: ["跳繩 6×30 秒（160–180 下/分）", "直膝彈跳 (pogo hops) 3×20，腳踝剛性、腳跟不落地", "A-skip 3×20 m、短坡衝刺 6×10 秒（慢走回復）", "赤足或薄底草地大步跑 4×60 m（每週 1 次）"], plan: "每週 2 次、各 10 分鐘，放在輕鬆跑後；長跑與比賽前 48 小時不做增強式訓練。" },
    dutyFactor: { title: "觸地佔比偏高", why: "觸地佔比高代表每個步幅週期大部分時間在地面上「支撐」而不是「飛行」，通常伴隨步頻低與腿部剛性不足。", cue: "提高腿部剛性：落地瞬間腳踝、膝蓋「硬」一點，不要往下沉。", drills: ["Pogo hops 3×20", "跳箱落地控制 3×6", "快腳 (fast feet) 3×10 秒"], plan: "與觸地時間一起改善。" },
    vo: { title: "垂直振幅偏大", why: `每一步都把體重往上舉：振幅 8 cm × 180 步/分 ≈ 每分鐘垂直位移 14 公尺，這些能量完全沒有幫你前進。垂直振幅偏大常與步頻低、著地點太前、以及「往上跑」的推蹬方式有關。`, cue: "「往前不是往上」、視線水平、感覺頭頂在一條水平線上滑行。",
      drills: ["步頻 +5% 節拍器跑（最有效）", "牆壁前傾 (wall drill) 3×30 秒：體會由腳踝前傾、靠重力前進", "核心穩定：棒式 3×45 秒、死蟲 3×10"], plan: "搭配步頻改善；若振幅已 < 6 cm 且步幅很短，反而要練步幅（髖伸展、大步跑）。" },
    vr: { title: "垂直比偏高（振幅相對步長）", why: "垂直比 = 垂直振幅 ÷ 步長，是跑步經濟性最直接的代理指標之一：同樣的垂直位移，步長越長代表這些上下動作「換到」越多前進距離。", cue: "把彈跳的方向從「上」改成「前」：髖部往前推、腳在身體下方落地。", drills: ["髖伸展 + 大步跑（見髖伸展項）", "步頻 +5% 節拍器跑"], plan: "追蹤 COROS 的垂直比：目標 < 7.5%。" },
    shankIC: { title: "過度跨步（著地點太前）", why: "著地時小腿角度越大，腳就越遠在重心前方：每一步都先「煞車」再重新加速（水平制動衝量↑），也會延長觸地時間、增加髕股關節與脛骨負荷（Souza 2016 建議小腿角 ≤ 5–7°）。這是業餘跑者最常見、也最值得改的經濟性漏洞。", cue: "「腳踝在膝蓋正下方落地」、「小腿垂直」；提高步頻是最簡單的修正手段。",
      drills: ["步頻 +5–10%（節拍器）", "牆壁前傾 drill：手撐牆、身體一直線前傾 20°，原地高抬腿 3×20", "快腳 drill 3×10 秒 + 15 m 加速轉換", "跑步機 2% 坡度輕鬆跑 10 分鐘（坡度自然縮短著地點）"], plan: "改善期間避免下坡衝刺；閾值課後段疲勞時最容易回到舊模式，拍一段課表後段的影片比對。" },
    kneeIC: { title: "著地時膝蓋太直", why: "膝蓋接近鎖直著地會把衝擊直接送到關節，吸震差、制動大；通常與過度跨步一起出現。", cue: "落地時膝蓋「軟」一點、腳在身體下方。", drills: ["步頻 +5%", "赤足草地慢跑 5 分鐘（自然縮短步幅）"], plan: "與過度跨步一起修正。" },
    footStrike: { title: "後足著地 + 跨步", why: "腳跟著地本身不是錯，菁英馬拉松選手也有 70% 以上是腳跟先著地；問題在於「腳跟著地 + 腳在身體很前方」= 制動。若小腿角小、著地點靠近身體，近端腳跟著地是可持續且經濟的。", cue: "不用刻意改成前足（突然轉前足會讓小腿／阿基里斯腱過載），先把著地點拉回身體下方。", drills: ["同過度跨步處方"], plan: "著地方式讓它隨步頻與姿勢自然改變，不要強迫。" },
    trunkLean: { title: "軀幹前傾不理想", why: "適度前傾（5–10°，由腳踝而非腰部）讓重力協助前進、髖伸展更順；太直立會用膝蓋抬腿、步幅靠前方；前傾過度或彎腰則壓迫呼吸、髖屈肌緊繃、腳跟著地更重。", cue: "「像一根棍子從腳踝倒下」、胸口打開、骨盆中立。",
      drills: ["牆壁前傾 3×30 秒", "落下起跑 (falling starts) 6×20 m：站直→前倒→自然跑出", "髖屈肌伸展 (couch stretch) 2×60 秒/側、臀橋 3×12"], plan: "疲勞時彎腰是常見的失速訊號；長跑最後 3 km 提醒自己「胸口向前」。" },
    hipExt: { title: "髖伸展不足（後擺太少）", why: "推進力主要來自支撐腿把身體「送出去」的髖伸展；髖伸展不足的跑者會用抬膝、往前跨來補步幅，造成過度跨步與垂直振幅。久坐族群的髖屈肌緊繃、臀肌啟動差是最常見原因。", cue: "「用臀部把地面往後推」、「腳在身體後方離地」。",
      drills: ["Couch stretch 2×60 秒/側（每天）", "臀橋 3×12、單腳 RDL 3×8/側（週 2 次）", "A-skip、B-skip 各 3×20 m", "短坡衝刺 6×10 秒（坡度 5–8%）：自然強迫髖伸展"], plan: "每週 2 次，與大步跑日合併。" },
    kneeSwing: { title: "後擺腳跟回收偏低", why: "擺動腿越直，鐘擺越長、擺動越慢，步頻會被卡住；提高腳跟回收高度縮短擺動腿的轉動慣量，是提升步頻而不增加費力感的關鍵。", cue: "「腳跟往臀部帶」而不是抬膝。", drills: ["Butt kicks 3×20 m", "快腳 + 腳跟回收 drill 3×15 秒", "大步跑 6×20 秒專注回收"], plan: "配速越快回收應越高；輕鬆跑 90° 左右即可。" },
    kneeStance: { title: "支撐期膝蓋塌陷或太僵硬", why: "支撐期膝彎曲太大代表腿部剛性不足（能量被肌肉吸收、不回彈）；太小則吸震不足。", cue: "落地「硬一點」但不要鎖死。", drills: ["Pogo hops、單腳跳 3×10/側", "深蹲／保加利亞分腿蹲 3×8"], plan: "增強式 + 力量訓練每週 2 次。" },
    elbow: { title: "手肘角度不理想", why: "手臂是腿部動作的平衡與節拍器：手臂太直（>110°）擺動慢、拖慢步頻；太緊（<70°）通常伴隨聳肩與上半身緊繃，浪費能量。", cue: "「手肘往後拉、手掌放鬆」、肩膀遠離耳朵。", drills: ["鏡子前原地擺臂 3×30 秒", "跑步中每 1 km 甩手放鬆肩膀 5 秒"], plan: "疲勞時手肘角度會變直、肩膀上提——比賽中用它當作放鬆提醒。" },
    armAmp: { title: "手臂擺幅不足或過大", why: "擺幅太小 = 手臂沒有參與節奏與平衡；過大 = 多餘的旋轉與能量浪費。", cue: "手從髖側擺到胸口下緣即可。", drills: ["鏡前擺臂 3×30 秒"], plan: "—" },
    headFwd: { title: "頭部前伸", why: "頭往前伸會帶動胸椎屈曲、呼吸變淺，並把重心往前帶而不是用前傾。", cue: "下巴微收、視線看前方 20–30 公尺。", drills: ["胸椎伸展（滾筒）2×10"], plan: "—" },
    trunkLeanSd: { title: "軀幹晃動偏大", why: "每一步前傾角度變化大代表核心沒有穩定住軀幹，推進力有部分被軀幹的來回擺動吃掉。", cue: "骨盆中立、肋骨「關起來」。", drills: ["棒式 3×45 秒、側橋 3×30 秒/側、死蟲 3×10"], plan: "每週 3 次、10 分鐘。" },
    gctAsym: { title: "左右觸地時間不對稱", why: "左右差 > 8% 通常代表單側力量／活動度差異或舊傷保護模式；長距離下弱側會先疲勞，是傷害與後段掉速的前兆。", cue: "感受弱側腳落地的聲音是否較重。", drills: ["單腳平衡 3×30 秒/側（閉眼進階）", "單腳 RDL、單腳跳 3×8/側", "弱側多做一組"], plan: "若連續 3 次分析都 > 10%，建議物理治療評估。" },
    pelvicDrop: { title: "骨盆下沉（臀中肌控制不足）", why: "支撐期對側骨盆下沉代表臀中肌／髖外展肌無法穩住骨盆，能量側漏、ITB 與膝關節負荷上升；Bramah (2018) 發現受傷跑者的骨盆下沉平均多 1°，而每增加 1° 受傷機率約提高 80%。", cue: "「骨盆像一碗水，不要灑出來」、支撐腳用力把骨盆撐高。",
      drills: ["側躺抬腿 3×15/側、蚌式 3×15/側", "單腳站立 hip hike 3×12/側", "側橋 3×30–45 秒/側", "單腳深蹲（鏡前）3×8/側"], plan: "每週 3 次，6–8 週後重新拍攝正面／背面影片。" },
    fppa: { title: "膝內扣（動態外翻）", why: "支撐期膝蓋往內塌陷 = 髖外旋／外展肌控制不足，髕股關節壓力與 ITB 張力上升，也浪費推進力。", cue: "「膝蓋對準第二腳趾」。", drills: ["彈力帶側走 3×15 步/方向", "單腳深蹲鏡前 3×8/側", "Step-down 控制 3×10/側"], plan: "與骨盆下沉一起練，6–8 週再評估。" },
    footLateral: { title: "交叉步（著地越過中線）", why: "腳落在身體中線的另一側 = 髖內收過多，拉緊 ITB、增加骨盆下沉，並讓每一步多一個橫向分量。", cue: "「沿著兩條平行線跑」，不是走一條鋼索。", drills: ["沿跑道線跑：兩腳分別落在線的兩側 4×200 m", "臀中肌訓練（同骨盆下沉）"], plan: "2–3 週意識練習通常就能修正。" },
    armCross: { title: "手臂過度越過中線", why: "手掌越過身體中線會帶動軀幹旋轉，骨盆跟著反向旋轉，製造不必要的扭轉與橫向力。", cue: "「手肘往後、拇指朝前」，手掌停在胸口下緣。", drills: ["鏡前原地擺臂 3×30 秒", "手持短棍跑 drill（兩手握棍兩端保持平行）"], plan: "—" },
    sway: { title: "骨盆側向擺動偏大", why: "每一步左右晃動是純粹的能量浪費，常和交叉步、臀中肌無力或步寬過窄有關。", cue: "想像頭頂一杯水、骨盆直線前進。", drills: ["側橋、農夫走路 (suitcase carry) 3×30 m/側", "臀中肌訓練"], plan: "—" },
    shoulderTilt: { title: "肩膀不水平", why: "可能是單側緊繃、持手機跑步或脊椎側彎的代償；長距離會變成單側負荷。", cue: "兩側肩膀同高、放鬆。", drills: ["胸椎／肩部活動度 5 分鐘"], plan: "—" },
  };

  function buildFindings(res, sc) {
    const issues = [], goods = [];
    for (const row of sc.rows) {
      const k = KNOWLEDGE[row.key]; if (!k || row.score === null || row.w === 0) continue;
      const deficit = (100 - row.score) * row.w;
      if (row.score < 78) issues.push({ ...k, key: row.key, score: row.score, deficit, observed: `${row.label} ${row.text}${row.unit ? " " + row.unit : ""}（目標 ${row.target}）`, severity: row.score < 55 ? "high" : "mid" });
      else if (row.score >= 90 && row.w >= 6) goods.push(`${row.label} ${row.text}${row.unit ? " " + row.unit : ""} — 達標`);
    }
    issues.sort((a, b) => b.deficit - a.deficit);
    // merge the overstride cluster so the advice is not repeated three times
    const cluster = ["shankIC", "kneeIC", "footStrike"]; let seen = false;
    const merged = []; for (const it of issues) { if (cluster.includes(it.key)) { if (seen) continue; seen = true; } merged.push(it); }
    return { issues: merged.slice(0, 4), goods, minor: merged.slice(4) };
  }

  /** economy framing for the HM goal: step length implied by goal pace × cadence */
  function goalTable(res, athlete) {
    const paces = [["A 目標 4:07", 247], ["B 目標 4:16", 256], ["閾值 T", null], ["此影片", res.pace_s]];
    const cads = [170, 176, 180, 184];
    const rows = paces.filter((p) => isNum(p[1])).map(([name, p]) => ({ name, p, cells: cads.map((c) => r1((1000 / p) / (c / 60) * 100) / 100) }));
    return { cads, rows, height: res.heightCm };
  }

  // ------------------------------------------------------------ COROS data
  const COROS = { index: null, cache: {} };
  async function loadIndex(base) {
    if (COROS.index) return COROS.index;
    const r = await fetch((base || "") + "dyn/index.json?t=" + Date.now(), { cache: "no-store" });
    if (!r.ok) throw new Error("index.json " + r.status);
    COROS.index = await r.json(); return COROS.index;
  }
  async function loadDyn(labelId, base) {
    if (COROS.cache[labelId]) return COROS.cache[labelId];
    const r = await fetch((base || "") + "dyn/" + labelId + ".json?t=" + Date.now(), { cache: "no-store" });
    if (!r.ok) throw new Error("dyn " + r.status);
    const d = await r.json(); COROS.cache[labelId] = d; return d;
  }
  /** candidates for a recording epoch (s): inside a workout, else nearest within 2 h; handles a ±8 h timezone mistake */
  function matchActivity(index, epoch) {
    const acts = index.activities || [];
    const inside = (e) => acts.filter((a) => a.start - 300 <= e && e <= a.end + 300);
    for (const [adj, note] of [[0, ""], [-8 * 3600, "影片時間被當成 UTC 解讀，已以 −8 h 修正"], [8 * 3600, "影片時間疑似寫成本地時間，已以 +8 h 修正"]]) {
      const e = epoch + adj; const c = inside(e);
      if (c.length) { const a = c.sort((p, q) => Math.abs(p.start - e) - Math.abs(q.start - e))[0]; return { act: a, epoch: e, offset: Math.max(0, e - a.start), confidence: "high", note }; }
    }
    const near = acts.map((a) => ({ a, d: Math.min(Math.abs(a.start - epoch), Math.abs(a.end - epoch)) })).filter((x) => x.d <= 2 * 3600).sort((p, q) => p.d - q.d);
    if (near.length) return { act: near[0].a, epoch, offset: 0, confidence: "low", note: `最近的訓練距離影片時間 ${Math.round(near[0].d / 60)} 分鐘，請確認` };
    return null;
  }
  function windowStats(dyn, t0, t1) {
    const C = {}; dyn.cols.forEach((c, i) => (C[c] = i));
    let rows = dyn.rows.filter((r) => r[0] >= t0 && r[0] <= t1);
    if (rows.length < 3) rows = dyn.rows.filter((r) => r[0] >= t0 - 3 && r[0] <= t1 + 3);
    const avg = (c) => { const i = C[c]; if (i === undefined) return null; const xs = rows.map((r) => r[i]).filter(isNum); return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null; };
    const v = avg("v_cms"); const speed = isNum(v) ? v / 100 : null;
    return { n: rows.length, t0, t1, speed: r1(speed * 100) / 100, pace_s: speed && speed > 0.5 ? Math.round(1000 / speed) : null, cad: r0(avg("cad")), sl: r0(avg("sl")), hr: r0(avg("hr")), vo: r1(avg("vo")), gct: r0(avg("gct")), vr: r1(avg("vr")), pw: r0(avg("pw")) };
  }
  function sessionStats(dyn) { const s = dyn.session || {}; return { speed: s.speed, pace_s: s.speed ? Math.round(1000 / s.speed) : null, cad: s.cad, sl: s.sl, hr: s.hr, vo: s.rec_vo ?? s.vo, gct: s.rec_gct ?? s.gct, vr: s.rec_vr ?? s.vr, pw: s.pw, dist_m: s.dist_m, time_s: s.time_s }; }
  /** ±90 s context series around the window for a small chart */
  function contextSeries(dyn, t0, t1) { const C = {}; dyn.cols.forEach((c, i) => (C[c] = i)); return dyn.rows.filter((r) => r[0] >= t0 - 90 && r[0] <= t1 + 90).map((r) => ({ t: r[0], cad: r[C.cad], gct: r[C.gct], pace: isNum(r[C.v_cms]) && r[C.v_cms] > 50 ? Math.round(100000 / r[C.v_cms]) : null, vo: r[C.vo] })); }

  // ------------------------------------------------------- video metadata
  /** Recording time from MP4/MOV boxes: Apple creationdate (with tz) > mvhd creation_time (UTC) > null */
  async function readCreationTime(file) {
    const rd = async (off, len) => new DataView(await file.slice(off, off + len).arrayBuffer());
    const type4 = (dv, o) => String.fromCharCode(dv.getUint8(o), dv.getUint8(o + 1), dv.getUint8(o + 2), dv.getUint8(o + 3));
    let off = 0, moov = null, guard = 0;
    while (off < file.size - 8 && guard++ < 64) {
      const dv = await rd(off, 16); let size = dv.getUint32(0); const type = type4(dv, 4); let hdr = 8;
      if (size === 1) { size = Number(dv.getBigUint64(8)); hdr = 16; } else if (size === 0) size = file.size - off;
      if (type === "moov") { moov = { off, size }; break; }
      if (size < 8) break; off += size;
    }
    if (!moov) return null;
    const buf = new Uint8Array(await file.slice(moov.off, moov.off + Math.min(moov.size, 6e6)).arrayBuffer());
    // 1) ISO date with timezone anywhere in moov (com.apple.quicktime.creationdate)
    let txt = ""; for (let i = 0; i < buf.length; i++) { const c = buf[i]; txt += c >= 32 && c < 127 ? String.fromCharCode(c) : " "; }
    const m = txt.match(/(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)([+-]\d{2}:?\d{2}|Z)/);
    if (m) { const iso = m[1] + (m[2].length === 5 ? m[2].slice(0, 3) + ":" + m[2].slice(3) : m[2]); const e = Date.parse(iso); if (isNum(e)) return { epoch: Math.round(e / 1000), source: "QuickTime creationdate（含時區）" }; }
    // 2) mvhd
    const dv = new DataView(buf.buffer); let p = 8;
    while (p < buf.length - 8) {
      let size = dv.getUint32(p); const type = type4(dv, p + 4); if (size < 8) break;
      if (type === "mvhd") { const ver = dv.getUint8(p + 8); const ct = ver === 1 ? Number(dv.getBigUint64(p + 12)) : dv.getUint32(p + 12); const epoch = ct - 2082844800; if (epoch > 1262304000) return { epoch, source: "MP4 mvhd creation_time（UTC）" }; return null; }
      p += size;
    }
    return null;
  }

  global.FORM = Object.assign(F, { METRICS, KNOWLEDGE, scoreResult, buildFindings, goalTable, loadIndex, loadDyn, matchActivity, windowStats, sessionStats, contextSeries, readCreationTime, pl, cadT, gctGood, paceStr, COROS });
})(typeof window !== "undefined" ? window : globalThis);


/* ============================================================================
 * Part 3 — browser UI (tab 跑姿分析). Everything below touches the DOM only when
 * renderFormTab() is called from index.html after the password gate is open.
 * ========================================================================== */
(function (global) {
  "use strict";
  const F = global.FORM;
  const isNum = (v) => typeof v === "number" && Number.isFinite(v);
  const h = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const q = (sel) => (typeof document !== "undefined" ? document.querySelector(sel) : null);
  const paceStr = F.paceStr;
  const MP_VERSION = "1.2.2";
  const MP_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
  const MODELS = { full: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task", lite: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task", heavy: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task" };
  const HIST_KEY = "hm_form_history_v1";
  const S = { file: null, url: null, meta: null, frames: null, res: null, score: null, match: null, dyn: null, win: null, landmarker: null, model: null, cancel: false, busy: false, thumbs: [], manualPace: null, indexErr: null };

  // --------------------------------------------------------------- storage
  const store = { get() { try { return JSON.parse(localStorage.getItem(HIST_KEY) || "[]"); } catch (e) { return []; } }, set(v) { try { localStorage.setItem(HIST_KEY, JSON.stringify(v)); } catch (e) { console.warn("history save failed", e); } } };

  // ------------------------------------------------------------ rendering
  function renderFormTab() {
    const el = q("#tab-form"); if (!el) return;
    if (S.rendered && S.file && el.innerHTML) { renderReviews(); return; } // keep the loaded video / results; only refresh coach reviews
    S.rendered = true;
    const athlete = (global.D && global.D.athlete) || {};
    el.innerHTML = `
    <div class="card p-5">
      <div class="flex flex-wrap items-center justify-between gap-2">
        <div><div class="font-black text-lg">🎥 跑姿分析（影片 → 生物力學 → 跑步經濟性） <button class="ml-2 px-2 py-0.5 rounded-lg bg-slate-200 text-slate-700 text-xs font-bold align-middle" onclick="if(typeof lockAgain==='function')lockAgain()">🔒 鎖定</button></div>
        <div class="text-sm text-slate-600 mt-1">上傳 10–20 秒跑步影片（跑步機或戶外、側面或正／背面）。姿勢估計（MediaPipe Pose）完全在你的瀏覽器內執行，影片不會上傳到任何伺服器；若影片拍攝時間落在某次 COROS 訓練內，會自動對上該秒數的 COROS 步頻／步幅／觸地時間／垂直振幅／配速／心率／功率做交叉比對。</div></div>
        <details class="text-xs text-slate-600 max-w-xl"><summary class="font-bold text-slate-800">📖 拍攝指南（點開）</summary>
          <ul class="list-disc pl-5 mt-1 space-y-1">
            <li><b>側面 90°</b>（最重要）：相機與跑向垂直、距離 4–6 m、高度約腰部、全身入鏡、固定不動（腳架／靠牆／請人持穩不追焦）。</li>
            <li><b>正面或背面</b>：看骨盆穩定、膝內扣、交叉步、手臂過中線（跑步機最方便）。</li>
            <li><b>60 fps 或慢動作 120／240 fps</b>：觸地時間的精度取決於影格率（30 fps ≈ ±25 ms）。慢動作倍率會自動判斷，也可手動指定。</li>
            <li><b>手錶先開跑再拍</b>：影片檔的拍攝時間會對到該次訓練的秒數；請在同一段配速穩定的路段拍攝（例如 T 配速一段、E 配速一段）。</li>
            <li>戶外：請人側拍一段直線（相機不平移）；跑步機：坡度 1%、拍攝時別看鏡頭。</li>
          </ul></details>
      </div>
    </div>
    <div class="card p-5" id="fm-upload">
      <div id="fm-drop" class="border-2 border-dashed border-slate-300 rounded-2xl p-8 text-center cursor-pointer hover:border-orange-400 hover:bg-orange-50 transition">
        <div class="text-3xl">📤</div><div class="font-bold mt-2">點擊或拖曳影片到這裡</div><div class="text-xs text-slate-500 mt-1">MP4 / MOV（iPhone、Android 皆可）· 建議 ≤ 60 秒</div>
        <input type="file" id="fm-file" accept="video/*" class="hidden">
      </div>
      <div id="fm-setup" class="hidden mt-4 grid lg:grid-cols-2 gap-4">
        <div>
          <video id="fm-video" class="w-full rounded-xl bg-black max-h-[420px]" playsinline muted controls></video>
          <div id="fm-meta" class="text-xs text-slate-600 mt-2"></div>
        </div>
        <div class="space-y-3 text-sm">
          <div class="grid grid-cols-2 gap-2">
            <label class="block"><span class="text-xs text-slate-500 font-bold">視角</span><select id="fm-view" class="w-full border rounded-lg px-2 py-1.5"><option value="auto">自動偵測</option><option value="side">側面</option><option value="front">正面</option><option value="back">背面</option></select></label>
            <label class="block"><span class="text-xs text-slate-500 font-bold">慢動作倍率</span><select id="fm-slow" class="w-full border rounded-lg px-2 py-1.5"><option value="auto">自動</option><option value="1">1×（一般）</option><option value="2">2×</option><option value="4">4×（120 fps 慢動作）</option><option value="8">8×（240 fps 慢動作）</option></select></label>
            <label class="block"><span class="text-xs text-slate-500 font-bold">分析區段（秒）</span><div class="flex items-center gap-1"><input id="fm-t0" type="number" min="0" step="0.5" value="0" class="w-full border rounded-lg px-2 py-1.5"><span>–</span><input id="fm-t1" type="number" min="1" step="0.5" value="20" class="w-full border rounded-lg px-2 py-1.5"></div></label>
            <label class="block"><span class="text-xs text-slate-500 font-bold">模型</span><select id="fm-model" class="w-full border rounded-lg px-2 py-1.5"><option value="full">標準（建議）</option><option value="heavy">高精度（慢）</option><option value="lite">快速</option></select></label>
            <label class="block"><span class="text-xs text-slate-500 font-bold">身高 cm</span><input id="fm-height" type="number" value="${h(athlete.height_cm || 171)}" class="w-full border rounded-lg px-2 py-1.5"></label>
            <label class="block"><span class="text-xs text-slate-500 font-bold">體重 kg</span><input id="fm-weight" type="number" step="0.1" value="${h(athlete.weight_kg || 61.5)}" class="w-full border rounded-lg px-2 py-1.5"></label>
          </div>
          <div class="bg-slate-50 rounded-xl p-3" id="fm-match"><div class="text-xs text-slate-500">COROS 配對：等待影片…</div></div>
          <div class="flex flex-wrap gap-2 items-center">
            <button id="fm-run" class="px-4 py-2 rounded-xl bg-slate-900 text-white font-bold">▶ 開始分析</button>
            <button id="fm-cancel" class="hidden px-4 py-2 rounded-xl bg-rose-600 text-white font-bold">■ 取消</button>
            <span id="fm-status" class="text-xs text-slate-500"></span>
          </div>
          <div id="fm-progress" class="hidden"><div class="h-2 bg-slate-200 rounded-full overflow-hidden"><div id="fm-bar" class="h-2 bg-orange-500" style="width:0%"></div></div><div class="text-xs text-slate-500 mt-1" id="fm-ptext"></div></div>
        </div>
      </div>
      <div id="fm-live" class="hidden mt-3"><canvas id="fm-canvas" class="w-full rounded-xl bg-black max-h-[480px]"></canvas></div>
    </div>
    <div id="fm-results"></div>
    <div class="card p-5" id="fm-reviews"></div>
    <div class="card p-5" id="fm-history"></div>`;
    bindUpload(); renderHistory(); renderReviews();
    F.loadIndex().then(() => { S.indexErr = null; if (S.meta) autoMatch(); }).catch((e) => { S.indexErr = e; const m = q("#fm-match"); if (m) m.innerHTML = `<div class="text-xs text-rose-600">COROS 索引載入失敗（${h(e.message)}）— 仍可分析影片，配速請手動輸入。</div>`; });
  }

  function bindUpload() {
    const drop = q("#fm-drop"), inp = q("#fm-file"); if (!drop || !inp) return;
    drop.onclick = () => inp.click();
    inp.onchange = () => { if (inp.files && inp.files[0]) onFile(inp.files[0]); };
    drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("bg-orange-50"); };
    drop.ondragleave = () => drop.classList.remove("bg-orange-50");
    drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("bg-orange-50"); const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]; if (f) onFile(f); };
    const run = q("#fm-run"); if (run) run.onclick = startAnalysis;
    const cancel = q("#fm-cancel"); if (cancel) cancel.onclick = () => { S.cancel = true; };
  }

  async function onFile(file) {
    if (!file.type.startsWith("video/") && !/\.(mp4|mov|m4v|webm)$/i.test(file.name)) { alert("請選擇影片檔"); return; }
    S.file = file; S.frames = null; S.res = null; S.thumbs = []; S.win = null; S.match = null;
    if (S.url) URL.revokeObjectURL(S.url); S.url = URL.createObjectURL(file);
    const v = q("#fm-video"); q("#fm-setup").classList.remove("hidden"); q("#fm-results").innerHTML = "";
    v.src = S.url; v.muted = true; v.playsInline = true;
    await new Promise((res, rej) => { v.onloadedmetadata = () => res(); v.onerror = () => rej(new Error("無法載入影片（格式不支援？iPhone HEVC 請改用「最相容」格式或用 Chrome/Safari 開啟）")); }).catch((e) => { q("#fm-meta").innerHTML = `<span class="text-rose-600">${h(e.message)}</span>`; });
    const dur = v.duration || 0; const W = v.videoWidth, H = v.videoHeight;
    q("#fm-t1").value = Math.min(dur, 20).toFixed(1); q("#fm-t0").value = "0";
    q("#fm-meta").innerHTML = `讀取中…`;
    const [fps, created] = await Promise.all([detectFps(v), F.readCreationTime(file).catch(() => null)]);
    const lastMod = file.lastModified ? Math.round(file.lastModified / 1000) : null;
    S.meta = { duration: dur, W, H, fps: fps || 30, fpsDetected: !!fps, created: created ? created.epoch : null, createdSource: created ? created.source : null, lastMod, name: file.name, size: file.size };
    const fmtT = (e) => (e ? new Date(e * 1000).toLocaleString("zh-TW", { timeZone: "Asia/Taipei", hour12: false }) : "—");
    q("#fm-meta").innerHTML = `<div class="grid grid-cols-2 gap-x-3 gap-y-0.5"><div>檔案</div><div class="truncate">${h(file.name)} · ${(file.size / 1e6).toFixed(1)} MB</div><div>長度／解析度</div><div>${dur.toFixed(1)} s · ${W}×${H}</div><div>影格率</div><div>${fps ? fps + " fps" : "無法偵測（假設 30）"}</div><div>拍攝時間</div><div>${created ? fmtT(created.epoch) + ` <span class="text-slate-400">(${h(created.source)})</span>` : `<span class="text-amber-600">檔案無拍攝時間</span>`}</div><div>檔案修改時間</div><div>${fmtT(lastMod)}</div></div>
      <label class="block mt-2"><span class="text-xs text-slate-500 font-bold">手動指定拍攝時間（台北）</span><input id="fm-created" type="datetime-local" step="1" class="border rounded-lg px-2 py-1 text-xs" value="${created ? toLocalInput(created.epoch) : (lastMod ? toLocalInput(lastMod) : "")}"></label>`;
    const ci = q("#fm-created"); if (ci) ci.onchange = () => { const e = fromLocalInput(ci.value); if (e) { S.meta.created = e; S.meta.createdSource = "手動"; autoMatch(); } };
    v.currentTime = 0;
    autoMatch();
  }
  const toLocalInput = (epoch) => { const d = new Date(epoch * 1000); const p = new Intl.DateTimeFormat("sv-SE", { timeZone: "Asia/Taipei", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }).format(d); return p.replace(" ", "T"); };
  const fromLocalInput = (s) => { if (!s) return null; const m = s.match(/(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2}))?/); if (!m) return null; return Math.round(Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4] - 8, +m[5], +(m[6] || 0)) / 1000); };

  async function detectFps(v) {
    if (!v.requestVideoFrameCallback) return null;
    const times = [];
    try {
      v.currentTime = 0.05; await v.play();
      await new Promise((res) => { let n = 0; const cb = (now, md) => { times.push(md.mediaTime); if (++n >= 45 || md.mediaTime > 1.6) res(); else v.requestVideoFrameCallback(cb); }; v.requestVideoFrameCallback(cb); setTimeout(res, 2500); });
    } catch (e) { /* autoplay blocked */ } finally { v.pause(); }
    const d = []; for (let i = 1; i < times.length; i++) if (times[i] > times[i - 1]) d.push(times[i] - times[i - 1]);
    if (d.length < 5) return null;
    const med = F.util.median(d); const fps = 1 / med; const common = [23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60, 100, 120, 240];
    let best = common[0]; for (const c of common) if (Math.abs(c - fps) < Math.abs(best - fps)) best = c;
    return Math.abs(best - fps) / fps < 0.08 ? Math.round(best * 100) / 100 : Math.round(fps * 10) / 10;
  }

  // ----------------------------------------------------------- COROS match
  function autoMatch() {
    const box = q("#fm-match"); if (!box || !S.meta) return;
    const idx = F.COROS.index;
    if (!idx) { box.innerHTML = `<div class="text-xs text-slate-500">COROS 索引載入中…</div>`; return; }
    const epoch = S.meta.created || S.meta.lastMod;
    S.match = epoch ? F.matchActivity(idx, epoch) : null;
    const acts = idx.activities.slice(0, 60);
    const opts = acts.map((a) => `<option value="${a.labelId}" ${S.match && S.match.act.labelId === a.labelId ? "selected" : ""}>${h(a.start_local || a.date)} · ${h(a.name)} · ${a.distance_km ?? "-"} km @${paceStr(a.pace_s)}${a.has_dyn ? "" : "（無逐秒資料）"}</option>`).join("");
    box.innerHTML = `<div class="flex flex-wrap items-center gap-2 text-xs"><span class="font-bold text-slate-700">COROS 配對</span>
      ${S.match ? `<span class="badge ${S.match.confidence === "high" ? "bg-emerald-100 text-emerald-700" : "bg-amber-100 text-amber-700"}">${S.match.confidence === "high" ? "✅ 自動對上" : "⚠ 不確定"}</span>` : `<span class="badge bg-slate-200 text-slate-600">無法自動配對</span>`}
      ${S.match && S.match.note ? `<span class="text-amber-700">${h(S.match.note)}</span>` : ""}</div>
      <select id="fm-act" class="w-full border rounded-lg px-2 py-1.5 mt-2 text-xs"><option value="">— 不配對（手動輸入配速）—</option>${opts}</select>
      <div class="grid grid-cols-3 gap-2 mt-2 text-xs items-end">
        <label>影片在訓練中的第幾秒<input id="fm-offset" type="number" step="1" class="w-full border rounded-lg px-2 py-1" value="${S.match ? Math.round(S.match.offset) : 0}"></label>
        <label>微調（秒）<input id="fm-shift" type="range" min="-120" max="120" step="1" value="0" class="w-full"></label>
        <label>手動配速 (m:ss/km)<input id="fm-pace" type="text" placeholder="例 4:10" class="w-full border rounded-lg px-2 py-1"></label>
      </div>
      <div id="fm-win" class="mt-2 text-xs"></div>`;
    const sel = q("#fm-act"), off = q("#fm-offset"), sh = q("#fm-shift"), pc = q("#fm-pace");
    const refresh = () => { S.manualPace = parsePace(pc.value); loadWindow(sel.value, Number(off.value) + Number(sh.value)); };
    sel.onchange = () => { const a = idx.activities.find((x) => x.labelId === sel.value); if (a && epoch) off.value = Math.max(0, Math.round(epoch - a.start)); refresh(); };
    off.onchange = refresh; sh.oninput = refresh; pc.onchange = refresh;
    refresh();
  }
  const parsePace = (s) => { const m = String(s || "").trim().match(/^(\d{1,2}):(\d{2})$/); return m ? +m[1] * 60 + +m[2] : null; };
  async function loadWindow(labelId, offset) {
    const box = q("#fm-win"); S.win = null; S.dyn = null;
    if (!labelId) { if (box) box.innerHTML = `<span class="text-slate-500">未配對：Morin 剛性模型與步長需要配速，請填手動配速（跑步機請換算 km/h → min/km）。</span>`; if (S.res) rerenderWithSpeed(); return; }
    try {
      const dyn = await F.loadDyn(labelId); S.dyn = dyn;
      const dur = S.meta ? (Number(q("#fm-t1").value) - Number(q("#fm-t0").value)) / (S.res ? S.res.slowmo : 1) : 20;
      const w = F.windowStats(dyn, offset, offset + Math.max(3, dur)); S.win = { ...w, labelId, offset };
      const ses = F.sessionStats(dyn);
      if (box) box.innerHTML = `<div class="grid grid-cols-4 md:grid-cols-8 gap-1 text-center">${[["配速", paceStr(w.pace_s)], ["步頻", w.cad ?? "-"], ["步長 cm", w.sl != null ? Math.round(w.sl / 10) : "-"], ["觸地 ms", w.gct ?? "-"], ["振幅 cm", w.vo != null ? (w.vo / 10).toFixed(1) : "-"], ["垂直比 %", w.vr ?? "-"], ["心率", w.hr ?? "-"], ["功率 W", w.pw ?? "-"]].map((x) => `<div class="bg-white rounded-lg p-1"><div class="text-[10px] text-slate-500">${x[0]}</div><div class="font-black font-mono">${x[1]}</div></div>`).join("")}</div>
        <div class="text-slate-500 mt-1">視窗 ${offset}–${Math.round(offset + Math.max(3, dur))} s（${w.n} 筆逐秒資料）· 全程平均：${paceStr(ses.pace_s)} · ${ses.cad ?? "-"} spm · GCT ${ses.gct ?? "-"} ms · 振幅 ${ses.vo != null ? (ses.vo / 10).toFixed(1) : "-"} cm</div>`;
      if (S.res) rerenderWithSpeed();
    } catch (e) { if (box) box.innerHTML = `<span class="text-rose-600">逐秒資料載入失敗（${h(e.message)}）— 可能是較舊的訓練（僅保留 90 天）或 FIT 尚未同步。</span>`; }
  }
  function currentSpeed() { if (S.manualPace) return 1000 / S.manualPace; if (S.win && S.win.speed) return S.win.speed; return null; }

  // ------------------------------------------------------------- MediaPipe
  async function getLandmarker(model) {
    if (S.landmarker && S.model === model) return S.landmarker;
    setStatus("載入姿勢模型（首次約 10–20 MB）…");
    const vision = await import(`${MP_BASE}/vision_bundle.mjs`);
    const fileset = await vision.FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
    const make = (delegate) => vision.PoseLandmarker.createFromOptions(fileset, { baseOptions: { modelAssetPath: MODELS[model] || MODELS.full, delegate }, runningMode: "VIDEO", numPoses: 1, minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5, minTrackingConfidence: 0.5, outputSegmentationMasks: false });
    try { S.landmarker = await make("GPU"); } catch (e) { console.warn("GPU delegate failed, falling back to CPU", e); S.landmarker = await make("CPU"); }
    S.model = model; S.tsLast = 0; return S.landmarker;
  }
  const seekTo = (v, t) => new Promise((res) => { let done = false; const fin = () => { if (!done) { done = true; v.removeEventListener("seeked", fin); res(); } }; v.addEventListener("seeked", fin); v.currentTime = t; setTimeout(fin, 600); });
  const nextFrame = () => new Promise((r) => (typeof requestAnimationFrame === "function" ? requestAnimationFrame(() => r()) : setTimeout(r, 0)));
  function setStatus(t) { const s = q("#fm-status"); if (s) s.textContent = t || ""; }

  async function startAnalysis() {
    if (S.busy || !S.file || !S.meta) return;
    S.busy = true; S.cancel = false; S.thumbs = [];
    const v = q("#fm-video"), cv = q("#fm-canvas"), ctx = cv.getContext("2d");
    const t0 = Math.max(0, Number(q("#fm-t0").value) || 0), t1 = Math.min(S.meta.duration, Number(q("#fm-t1").value) || S.meta.duration);
    if (t1 - t0 < 3) { alert("分析區段至少 3 秒"); S.busy = false; return; }
    if (t1 - t0 > 60 && !confirm("區段超過 60 秒，處理時間會很長，確定繼續？")) { S.busy = false; return; }
    q("#fm-run").classList.add("hidden"); q("#fm-cancel").classList.remove("hidden"); q("#fm-progress").classList.remove("hidden"); q("#fm-live").classList.remove("hidden"); q("#fm-results").innerHTML = "";
    try {
      const lm = await getLandmarker(q("#fm-model").value);
      v.pause(); const fps = S.meta.fps; const W = S.meta.W, H = S.meta.H;
      const scale = Math.min(1, 640 / Math.max(W, H)); cv.width = Math.round(W * scale); cv.height = Math.round(H * scale);
      const frames = []; const n = Math.floor((t1 - t0) * fps); const started = Date.now();
      const tsBase = (S.tsLast || 0) + 1000; // MediaPipe VIDEO mode requires strictly increasing timestamps per instance
      for (let i = 0; i < n; i++) {
        if (S.cancel) throw new Error("已取消");
        const t = t0 + (i + 0.5) / fps; await seekTo(v, t);
        let r = null; try { r = lm.detectForVideo(v, tsBase + Math.round(i * 1000 / fps)); } catch (e) { console.warn(e); }
        const pose = r && r.landmarks && r.landmarks[0] ? r.landmarks[0] : null;
        frames.push({ t: v.currentTime || t, lm: pose });
        if (i % 2 === 0 || i === n - 1) { ctx.drawImage(v, 0, 0, cv.width, cv.height); if (pose) drawSkeleton(ctx, pose, cv.width, cv.height); const pct = Math.round((i + 1) / n * 100); q("#fm-bar").style.width = pct + "%"; const el = (Date.now() - started) / 1000; q("#fm-ptext").textContent = `${i + 1}/${n} 影格 · ${pct}% · 已用 ${el.toFixed(0)} s · 預估剩餘 ${(el / (i + 1) * (n - i - 1)).toFixed(0)} s`; await nextFrame(); }
      }
      S.tsLast = tsBase + Math.round(n * 1000 / fps) + 1;
      setStatus("計算指標…"); await nextFrame();
      const valid = frames.filter((f) => f.lm);
      if (valid.length < 10) throw new Error("幾乎沒有偵測到人體，請確認全身入鏡、光線足夠、且影片方向正確");
      const refCad = S.win && S.win.cad ? S.win.cad : null;
      const speed = currentSpeed();
      const res = F.analyzeFrames(frames.filter((f) => f.lm), { W, H, fps, heightCm: Number(q("#fm-height").value) || 171, weightKg: Number(q("#fm-weight").value) || 61.5, slowmo: q("#fm-slow").value, view: q("#fm-view").value, refCadence: refCad, speed });
      res.detectRate = Math.round(valid.length / frames.length * 100);
      S.frames = frames; S.res = res;
      if (S.win && S.win.labelId) await loadWindow(S.win.labelId, S.win.offset); // window length depends on slow-mo factor
      S.score = F.scoreResult(res, { pace_s: speed ? Math.round(1000 / speed) : null });
      setStatus("擷取關鍵影格…"); await nextFrame();
      S.thumbs = await captureKeyFrames(v, res, frames, W, H);
      renderResults(); saveToHistory(); setStatus("完成 ✓");
    } catch (e) {
      console.error(e); q("#fm-results").innerHTML = `<div class="card p-5 text-rose-600">分析失敗：${h(e.message)}</div>`; setStatus("");
    } finally {
      S.busy = false; q("#fm-run").classList.remove("hidden"); q("#fm-cancel").classList.add("hidden"); q("#fm-progress").classList.add("hidden");
    }
  }
  function rerenderWithSpeed() { if (!S.res) return; const speed = currentSpeed(); if (speed) Object.assign(S.res, F.speedMetrics(S.res, speed, S.res.heightCm, S.res.weightKg)); else for (const k of ["speed", "pace_s", "stepLen", "vr", "stepLenPctHeight", "Fmax_bw", "comDrop", "kvert", "kleg", "vertLiftPerMin"]) delete S.res[k]; S.score = F.scoreResult(S.res, { pace_s: speed ? Math.round(1000 / speed) : null }); renderResults(); }

  function drawSkeleton(ctx, pose, W, H, opts = {}) {
    ctx.lineWidth = opts.lw || 3; ctx.strokeStyle = opts.color || "rgba(249,115,22,.9)"; ctx.fillStyle = opts.color || "rgba(249,115,22,.95)";
    for (const [a, b] of F.CONNECTIONS) { const p = pose[a], r = pose[b]; if (!p || !r || (p.visibility ?? 1) < 0.3 || (r.visibility ?? 1) < 0.3) continue; ctx.beginPath(); ctx.moveTo(p.x * W, p.y * H); ctx.lineTo(r.x * W, r.y * H); ctx.stroke(); }
    for (let i = 11; i < 33; i++) { const p = pose[i]; if (!p || (p.visibility ?? 1) < 0.3) continue; ctx.beginPath(); ctx.arc(p.x * W, p.y * H, opts.lw ? opts.lw + 1 : 4, 0, Math.PI * 2); ctx.fill(); }
  }

  // ------------------------------------------------------------ key frames
  async function captureKeyFrames(v, res, frames, W, H) {
    const out = []; const ev = res.events || []; if (!ev.length) return out;
    const side = res.view === "side" ? res.nearSide : "L"; const other = side === "L" ? "R" : "L";
    const mine = ev.filter((e) => e.side === side); const e = mine[Math.floor(mine.length / 2)] || ev[0];
    const oth = ev.filter((e2) => e2.side === other); const e2 = oth[Math.floor(oth.length / 2)];
    const list = [["著地 IC", e.icVideoT, side], ["支撐中期", e.lowVideoT, side], ["離地 TO", e.toVideoT, side]]; if (e2) list.push(["著地 IC（另一腳）", e2.icVideoT, other]);
    const cv = document.createElement("canvas"); const scale = Math.min(1, 520 / Math.max(W, H)); cv.width = Math.round(W * scale); cv.height = Math.round(H * scale); const ctx = cv.getContext("2d");
    for (const [label, t, sd] of list) {
      if (!isNum(t)) continue;
      await seekTo(v, t);
      ctx.drawImage(v, 0, 0, cv.width, cv.height);
      let best = null, bd = Infinity; for (const f of frames) { if (!f.lm) continue; const d = Math.abs(f.t - t); if (d < bd) { bd = d; best = f; } }
      if (best) { drawSkeleton(ctx, best.lm, cv.width, cv.height, { lw: 2 }); annotate(ctx, best.lm, cv.width, cv.height, sd, res, label); }
      ctx.fillStyle = "rgba(15,23,42,.75)"; ctx.fillRect(0, 0, cv.width, 22); ctx.fillStyle = "#fff"; ctx.font = "bold 13px sans-serif"; ctx.fillText(`${label} · ${sd === "L" ? "左" : "右"}腳 · t=${t.toFixed(2)}s`, 6, 15);
      out.push({ label, t, side: sd, dataUrl: cv.toDataURL("image/jpeg", 0.72) });
    }
    return out;
  }
  function annotate(ctx, lm, W, H, side, res, label) {
    const Sd = F.SIDES[side]; const P = (k) => ({ x: lm[k].x * W, y: lm[k].y * H });
    const deg = (a, b, c) => { const v1x = a.x - b.x, v1y = a.y - b.y, v2x = c.x - b.x, v2y = c.y - b.y; const d = Math.hypot(v1x, v1y) * Math.hypot(v2x, v2y); return d ? Math.acos(Math.max(-1, Math.min(1, (v1x * v2x + v1y * v2y) / d))) * 180 / Math.PI : 0; };
    ctx.font = "bold 12px sans-serif"; ctx.fillStyle = "#fde68a"; ctx.strokeStyle = "rgba(0,0,0,.6)"; ctx.lineWidth = 3;
    const txt = (s, p) => { ctx.strokeText(s, p.x + 6, p.y - 6); ctx.fillText(s, p.x + 6, p.y - 6); };
    const hip = P(Sd.hip), knee = P(Sd.knee), ank = P(Sd.ank), sh = P(Sd.sh);
    txt(`膝 ${Math.round(180 - deg(hip, knee, ank))}°`, knee);
    if (res.view === "side") {
      const dir = res.dir; const shank = Math.atan2(dir * (ank.x - knee.x), ank.y - knee.y) * 180 / Math.PI; txt(`小腿 ${shank.toFixed(0)}°`, ank);
      const mh = { x: (P(23).x + P(24).x) / 2, y: (P(23).y + P(24).y) / 2 }, ms = { x: (P(11).x + P(12).x) / 2, y: (P(11).y + P(12).y) / 2 };
      const lean = Math.atan2(dir * (ms.x - mh.x), mh.y - ms.y) * 180 / Math.PI; txt(`前傾 ${lean.toFixed(0)}°`, ms);
      // vertical reference through the pelvis
      ctx.setLineDash([4, 4]); ctx.strokeStyle = "rgba(255,255,255,.6)"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(mh.x, 0); ctx.lineTo(mh.x, H); ctx.stroke(); ctx.setLineDash([]);
    } else {
      const hl = P(23), hr = P(24); ctx.strokeStyle = "rgba(255,255,255,.8)"; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(hl.x, hl.y); ctx.lineTo(hr.x, hr.y); ctx.stroke();
      const tilt = Math.atan2(hr.y - hl.y, Math.abs(hr.x - hl.x)) * 180 / Math.PI; txt(`骨盆 ${Math.abs(tilt).toFixed(1)}°`, { x: (hl.x + hr.x) / 2, y: hl.y });
    }
  }

  // ----------------------------------------------------------- results UI
  function scoreColor(s) { return s === null ? "text-slate-400" : s >= 85 ? "text-emerald-600" : s >= 70 ? "text-amber-600" : "text-rose-600"; }
  function barColor(s) { return s === null ? "#cbd5e1" : s >= 85 ? "#10b981" : s >= 70 ? "#f59e0b" : "#ef4444"; }
  function renderResults() {
    const el = q("#fm-results"); if (!el || !S.res) return;
    const r = S.res, sc = S.score, w = S.win, D = global.D || {};
    const fd = F.buildFindings(r, sc);
    const viewLabel = { side: "側面", front: "正面", back: "背面" }[r.view] || r.view;
    const kpi = (label, val, sub, color = "") => `<div class="bg-slate-50 rounded-xl p-3"><div class="text-[11px] text-slate-500 font-bold">${label}</div><div class="text-xl font-black font-mono ${color}">${val}</div><div class="text-[11px] text-slate-500">${sub || ""}</div></div>`;
    const cmp = (vid, cor, unit, tolPct) => { if (!isNum(vid) || !isNum(cor)) return ""; const d = (vid - cor) / cor * 100; const ok = Math.abs(d) <= tolPct; return `<span class="badge ${ok ? "bg-emerald-100 text-emerald-700" : "bg-amber-100 text-amber-700"}">COROS ${cor}${unit} ${ok ? "✓ 一致" : (d > 0 ? "▲" : "▼") + Math.abs(d).toFixed(0) + "%"}</span>`; };
    const pace = r.pace_s ? paceStr(r.pace_s) : null;
    const ring = sc.total === null ? "" : `<div class="relative w-28 h-28"><svg viewBox="0 0 36 36" class="w-28 h-28 -rotate-90"><circle cx="18" cy="18" r="15.9" fill="none" stroke="#e2e8f0" stroke-width="3.5"/><circle cx="18" cy="18" r="15.9" fill="none" stroke="${barColor(sc.total)}" stroke-width="3.5" stroke-dasharray="${sc.total} 100" stroke-linecap="round"/></svg><div class="absolute inset-0 flex flex-col items-center justify-center"><div class="text-3xl font-black">${sc.total}</div><div class="text-xs font-bold text-slate-500">${sc.grade}</div></div></div>`;
    const top = fd.issues[0];
    const summary = r.walking ? "這段影片是走路／無騰空期，跑姿評分不適用。" : top ? `最值得先改的是「${top.title}」：${top.cue}` : "整體跑姿很經濟，維持目前模式並持續追蹤。";
    el.innerHTML = `
    <div class="card p-5 space-y-5 fade-in">
      <div class="flex flex-wrap items-center gap-5">
        ${ring}
        <div class="flex-1 min-w-[240px]">
          <div class="flex flex-wrap gap-2 text-xs"><span class="badge bg-slate-900 text-white">${viewLabel}視角</span><span class="badge bg-slate-100 text-slate-700">${r.fps} fps${r.slowmo > 1 ? ` · 慢動作 ${r.slowmo}×` : ""}</span><span class="badge bg-slate-100 text-slate-700">${r.nContacts} 次著地 · ${r.duration}s</span><span class="badge bg-slate-100 text-slate-700">偵測率 ${r.detectRate ?? r.quality}%</span>${pace ? `<span class="badge bg-orange-100 text-orange-700">配速 ${pace}/km${w ? "（COROS）" : "（手動）"}</span>` : `<span class="badge bg-amber-100 text-amber-700">未知配速：以輕鬆跑標準評分</span>`}</div>
          <div class="font-black text-lg mt-2">${h(summary)}</div>
          ${r.warnings.length ? `<ul class="text-xs text-amber-700 mt-1 list-disc pl-4">${r.warnings.map((x) => `<li>${h(x)}</li>`).join("")}</ul>` : ""}
        </div>
      </div>
      <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-2">
        ${kpi("步頻 spm", r.cadence ?? "-", cmp(r.cadence, w && w.cad, "", 3), scoreColor(rowScore(sc, "cadence")))}
        ${kpi("觸地時間 ms", r.gct != null ? `${r.gct}<span class="text-xs text-slate-400"> ±${r.gctTol}</span>` : "-", (w && w.gct ? cmp(r.gct, w.gct, "", 12) : "") + (r.gctZeni ? `<div>兩法 ${r.gctPlateau}/${r.gctZeni}</div>` : ""), scoreColor(rowScore(sc, "gct")))}
        ${kpi("騰空時間 ms", r.flight ?? "-", r.dutyFactor != null ? `觸地佔比 ${r.dutyFactor}` : "", scoreColor(rowScore(sc, "dutyFactor")))}
        ${kpi("垂直振幅 cm", r.vo ?? "-", w && w.vo != null ? cmp(r.vo, Math.round(w.vo) / 10, "", 20) : "", scoreColor(rowScore(sc, "vo")))}
        ${kpi("垂直比 %", r.vr ?? (w && w.vr != null ? w.vr : "-"), w && w.vr != null ? (r.vr != null ? cmp(r.vr, w.vr, "", 20) : "COROS") : "需配速", scoreColor(rowScore(sc, "vr")))}
        ${kpi("步長 m", r.stepLen ?? (w && w.sl ? (w.sl / 1000).toFixed(2) : "-"), r.stepLenPctHeight ? `身高 ${r.stepLenPctHeight}%` : "", "")}
        ${r.view === "side" ? kpi("著地小腿角 °", r.shankIC ?? "-", r.footAhead != null ? `腳在骨盆前 ${r.footAhead} cm` : "", scoreColor(rowScore(sc, "shankIC"))) : kpi("骨盆下沉 °", r.pelvicDrop ?? "-", `L ${r.perSide?.L?.pelvicDrop ?? "-"} / R ${r.perSide?.R?.pelvicDrop ?? "-"}`, scoreColor(rowScore(sc, "pelvicDrop")))}
        ${r.view === "side" ? kpi("著地方式", ({ heel: "後足", midfoot: "中足", forefoot: "前足" })[r.footStrike] || "-", r.footIC != null ? `足角 ${r.footIC}° · 膝 ${r.kneeIC}°` : "", scoreColor(rowScore(sc, "footStrike"))) : kpi("膝內扣 FPPA °", r.fppa ?? "-", `L ${r.perSide?.L?.fppa ?? "-"} / R ${r.perSide?.R?.fppa ?? "-"}`, scoreColor(rowScore(sc, "fppa")))}
        ${r.view === "side" ? kpi("軀幹前傾 °", r.trunkLean ?? "-", `波動 ±${r.trunkLeanSd ?? "-"}°`, scoreColor(rowScore(sc, "trunkLean"))) : kpi("著地橫向 cm", r.footLateral ?? "-", `步寬 ${r.stepWidth ?? "-"} cm`, scoreColor(rowScore(sc, "footLateral")))}
        ${r.view === "side" ? kpi("髖伸展 (TO) °", r.hipExt ?? "-", `髖角 ${r.hipExtTO ?? "-"}°`, scoreColor(rowScore(sc, "hipExt"))) : kpi("手臂過中線 %", r.armCross ?? "-", `側擺 ${r.sway ?? "-"} cm`, scoreColor(rowScore(sc, "armCross")))}
        ${r.view === "side" ? kpi("後擺膝折疊 °", r.kneeSwing ?? "-", `支撐期最大 ${r.kneeStance ?? "-"}°`, scoreColor(rowScore(sc, "kneeSwing"))) : kpi("肩水平 °", r.shoulderTilt ?? "-", "", scoreColor(rowScore(sc, "shoulderTilt")))}
        ${kpi("左右對稱 (GCT)", r.gctAsym != null ? `${r.gctAsym}%` : "-", `L ${r.gctL ?? "-"} / R ${r.gctR ?? "-"} ms`, scoreColor(rowScore(sc, "gctAsym")))}
      </div>
      ${S.thumbs.length ? `<div><div class="font-bold text-sm mb-2">關鍵影格（${r.view === "side" ? (r.nearSide === "L" ? "靠鏡頭側＝左腳" : "靠鏡頭側＝右腳") : "正面／背面"}）</div><div class="grid grid-cols-2 md:grid-cols-4 gap-2">${S.thumbs.map((t) => `<div><img src="${t.dataUrl}" class="rounded-xl w-full border"><div class="text-[11px] text-slate-500 mt-1">${h(t.label)} · t=${t.t.toFixed(2)}s</div></div>`).join("")}</div></div>` : ""}
      <div class="grid md:grid-cols-2 gap-4">
        <div><div class="font-bold text-sm mb-1">每一步的觸地時間（L／R）</div><canvas id="fm-ch-gct" height="150"></canvas></div>
        <div><div class="font-bold text-sm mb-1">${r.view === "side" && r.perSide && (r.perSide.L.kneeCurve.length || r.perSide.R.kneeCurve.length) ? "膝關節彎曲角度（一個步幅週期，0% = 著地）" : "每一步垂直振幅"}</div><canvas id="fm-ch-knee" height="150"></canvas></div>
      </div>
      ${w && S.dyn ? `<div><div class="font-bold text-sm mb-1">COROS 同時段（灰底 = 影片視窗）：步頻／觸地時間／配速</div><canvas id="fm-ch-coros" height="150"></canvas></div>` : ""}
    </div>
    <div class="card p-5 space-y-4 fade-in">
      <div class="font-black text-lg">🧑‍🏫 教練解析：對半馬跑步經濟性的意義</div>
      ${r.walking ? `<div class="text-sm text-slate-600">請上傳跑步片段。</div>` : fd.issues.length ? fd.issues.map((it, i) => `<div class="rounded-2xl border ${it.severity === "high" ? "border-rose-200 bg-rose-50" : "border-amber-200 bg-amber-50"} p-4">
          <div class="flex flex-wrap items-center gap-2"><span class="badge ${it.severity === "high" ? "bg-rose-600 text-white" : "bg-amber-500 text-white"}">優先 ${i + 1}</span><span class="font-black">${h(it.title)}</span><span class="text-xs text-slate-500">得分 ${it.score}</span></div>
          <div class="text-sm mt-1"><b>觀察：</b>${h(it.observed)}</div>
          <div class="text-sm mt-1 text-slate-700"><b>為什麼影響經濟性：</b>${h(it.why)}</div>
          <div class="text-sm mt-1"><b>技術提示：</b>${h(it.cue)}</div>
          <div class="text-sm mt-1"><b>訓練處方：</b><ul class="list-disc pl-5">${it.drills.map((d) => `<li>${h(d)}</li>`).join("")}</ul></div>
          ${it.plan && it.plan !== "—" ? `<div class="text-xs mt-1 text-slate-600"><b>放進課表：</b>${h(it.plan)}</div>` : ""}
        </div>`).join("") : `<div class="text-sm text-emerald-700">沒有明顯的經濟性漏洞。</div>`}
      ${fd.goods.length ? `<div class="rounded-2xl border border-emerald-200 bg-emerald-50 p-4 text-sm"><b>✅ 做得好：</b><ul class="list-disc pl-5 mt-1">${fd.goods.map((g) => `<li>${h(g)}</li>`).join("")}</ul></div>` : ""}
      ${fd.minor.length ? `<div class="text-xs text-slate-600"><b>次要：</b>${fd.minor.map((m) => h(m.title)).join("、")}</div>` : ""}
      ${economyBlock(r, w, D)}
      <details class="text-sm"><summary class="font-bold">📋 全部指標與評分（${sc.rows.length} 項）</summary>
        <table class="w-full text-xs mt-2"><thead><tr class="text-left text-slate-500"><th class="py-1">指標</th><th>數值</th><th>目標</th><th>評分</th></tr></thead><tbody>
        ${sc.rows.map((x) => `<tr class="border-t"><td class="py-1">${h(x.label)}</td><td class="font-mono">${h(x.text)} ${h(x.unit)}</td><td class="text-slate-500">${h(x.target)}</td><td><div class="flex items-center gap-1"><div class="h-2 w-16 bg-slate-200 rounded"><div class="h-2 rounded" style="width:${x.score ?? 0}%;background:${barColor(x.score)}"></div></div><span class="${scoreColor(x.score)} font-bold">${x.score ?? "—"}</span></div></td></tr>`).join("")}
        </tbody></table>
        <div class="text-[11px] text-slate-500 mt-2">評分基準：步頻／觸地時間／後擺膝折疊依配速調整（Daniels 配速越快標準越高）；垂直振幅與垂直比參考 Garmin／COROS 跑者分布；過度跨步採小腿角 (Souza 2016)；著地方式採 Altman & Davis (2012) 足角分類；骨盆下沉參考 Bramah (2018)。影像法的角度誤差約 ±3°，觸地時間 ±1 影格。</div>
      </details>
      <div class="flex flex-wrap gap-2"><button class="px-3 py-1.5 rounded-lg bg-slate-900 text-white text-xs font-bold" onclick="FORM.copySummary()">📋 複製教練摘要（貼到教練室 Chat）</button><button class="px-3 py-1.5 rounded-lg bg-slate-200 text-slate-800 text-xs font-bold" onclick="FORM.downloadJson()">⬇ 下載分析 JSON</button><span id="fm-copied" class="text-xs text-emerald-600"></span></div>
    </div>`;
    drawCharts(r, w);
  }
  function rowScore(sc, key) { const row = sc.rows.find((x) => x.key === key); return row ? row.score : null; }

  function economyBlock(r, w, D) {
    const gt = F.goalTable(r, D.athlete || {});
    const stiff = r.kvert ? `<div class="grid grid-cols-2 md:grid-cols-5 gap-2 text-center text-xs mt-2">${[["峰值地面反作用力", `${r.Fmax_bw} BW`], ["支撐期重心下沉", `${r.comDrop} cm`], ["垂直剛性 k_vert", `${r.kvert} kN/m`], ["腿部剛性 k_leg", `${r.kleg} kN/m`], ["每分鐘垂直位移", `${r.vertLiftPerMin} m`]].map((x) => `<div class="bg-white rounded-lg p-2 border"><div class="text-[10px] text-slate-500">${x[0]}</div><div class="font-black font-mono">${x[1]}</div></div>`).join("")}</div>
      <div class="text-[11px] text-slate-500 mt-1">Morin et al. (2005) 彈簧－質量模型，由觸地／騰空時間、體重、速度推估；訓練有素的跑者 k_vert 約 20–35 kN/m、k_leg 8–14 kN/m。剛性提高（觸地縮短、重心下沉減少）與跑步經濟性改善相關，但過高伴隨衝擊負荷上升。數值隨影格率誤差放大，請看趨勢而非單次。</div>` : `<div class="text-xs text-slate-500 mt-1">（配對 COROS 或輸入配速後，會顯示 Morin 彈簧－質量模型的垂直／腿部剛性）</div>`;
    const cad = r.cadence || 176;
    return `<div class="rounded-2xl border border-sky-200 bg-sky-50 p-4 text-sm">
      <div class="font-bold">🎯 跑步經濟性與半馬目標（${(D.race && D.race.name) || "半馬"} A ${(D.race && D.race.goal_A) || "1:27"} / B ${(D.race && D.race.goal_B) || "1:30"}）</div>
      <div class="text-slate-700 mt-1">速度 = 步頻 × 步長。以你目前的步頻 ${cad} spm，半馬 A 目標 4:07/km 需要步長 ${((1000 / 247) / (cad / 60)).toFixed(2)} m（身高 ${Math.round((1000 / 247) / (cad / 60) * 100 / r.heightCm * 100)}%）；把步頻提到 180–184 可以把同速度的步長縮短 3–5%，直接減少著地點前移與垂直振幅，這通常是業餘跑者最便宜的經濟性改善。</div>
      <table class="text-xs mt-2 w-full"><thead><tr class="text-slate-500"><th class="text-left">配速</th>${gt.cads.map((c) => `<th>${c} spm</th>`).join("")}</tr></thead><tbody>${gt.rows.map((row) => `<tr class="border-t"><td class="py-0.5">${h(row.name)} ${paceStr(row.p)}</td>${row.cells.map((c) => `<td class="text-center font-mono">${c} m</td>`).join("")}</tr>`).join("")}</tbody></table>
      ${w ? `<div class="text-xs text-slate-600 mt-2">COROS 同時段：${paceStr(w.pace_s)}/km · 步頻 ${w.cad ?? "-"} · 步長 ${w.sl ? (w.sl / 1000).toFixed(2) : "-"} m · 觸地 ${w.gct ?? "-"} ms · 振幅 ${w.vo != null ? (w.vo / 10).toFixed(1) : "-"} cm · 垂直比 ${w.vr ?? "-"}% · 心率 ${w.hr ?? "-"} · 功率 ${w.pw ?? "-"} W${w.pw && w.speed ? ` · 功率／速度 ${(w.pw / w.speed).toFixed(0)} W per m/s（越低越經濟）` : ""}</div>` : ""}
      ${stiff}
    </div>`;
  }

  function drawCharts(r, w) {
    if (typeof Chart === "undefined" || typeof global.mkChart !== "function") return;
    const ev = r.events || [];
    global.mkChart("fm-ch-gct", { type: "bar", data: { labels: ev.map((e, i) => `${i + 1}${e.side}`), datasets: [{ label: "GCT ms", data: ev.map((e) => e.gct), backgroundColor: ev.map((e) => (e.side === "L" ? "#3b82f6" : "#f97316")) }] }, options: { plugins: { legend: { display: false } }, scales: { y: { suggestedMin: 150, title: { display: true, text: "ms" } } } } });
    if (r.view === "side" && r.perSide && (r.perSide.L.kneeCurve.length || r.perSide.R.kneeCurve.length)) {
      const ds = []; for (const s of ["L", "R"]) { const c = r.perSide[s].kneeCurve || []; if (c.length) ds.push({ label: `${s === "L" ? "左" : "右"}膝 (支撐 ${r.perSide[s].stancePct ?? "?"}%)`, data: c.map((p) => ({ x: p.p, y: p.k })), borderColor: s === "L" ? "#3b82f6" : "#f97316", pointRadius: 0, tension: 0.3 }); }
      global.mkChart("fm-ch-knee", { type: "line", data: { datasets: ds }, options: { parsing: false, scales: { x: { type: "linear", min: 0, max: 100, title: { display: true, text: "% 步幅週期" } }, y: { title: { display: true, text: "膝彎曲 °" } } } } });
    } else {
      global.mkChart("fm-ch-knee", { type: "line", data: { labels: (r.voList || []).map((_, i) => i + 1), datasets: [{ label: "VO cm", data: r.voList || [], borderColor: "#a855f7", pointRadius: 2, tension: 0.3 }] }, options: { plugins: { legend: { display: false } }, scales: { y: { title: { display: true, text: "cm" } } } } });
    }
    if (w && S.dyn && q("#fm-ch-coros")) {
      const cs = F.contextSeries(S.dyn, w.t0, w.t1);
      const shade = { id: "fmShade", beforeDraw(chart) { const { ctx, chartArea: a, scales: { x } } = chart; if (!a) return; const x0 = x.getPixelForValue(w.t0), x1 = x.getPixelForValue(w.t1); ctx.save(); ctx.fillStyle = "rgba(148,163,184,.25)"; ctx.fillRect(x0, a.top, Math.max(2, x1 - x0), a.bottom - a.top); ctx.restore(); } };
      global.mkChart("fm-ch-coros", { type: "line", data: { datasets: [{ label: "步頻", data: cs.map((p) => ({ x: p.t, y: p.cad })), borderColor: "#22c55e", pointRadius: 0, yAxisID: "y" }, { label: "GCT ms", data: cs.map((p) => ({ x: p.t, y: p.gct })), borderColor: "#f97316", pointRadius: 0, yAxisID: "y1" }, { label: "配速 s/km", data: cs.map((p) => ({ x: p.t, y: p.pace })), borderColor: "#64748b", borderDash: [4, 3], pointRadius: 0, yAxisID: "y2" }] }, options: { parsing: false, scales: { x: { type: "linear", title: { display: true, text: "訓練第 n 秒" } }, y: { position: "left", title: { display: true, text: "spm" } }, y1: { position: "right", grid: { drawOnChartArea: false }, title: { display: true, text: "ms" } }, y2: { display: false, reverse: true } } }, plugins: [shade] });
    }
  }

  // --------------------------------------------------------------- history
  function saveToHistory() {
    const r = S.res, sc = S.score; if (!r || !sc) return;
    const entry = { id: Date.now(), ts: new Date().toISOString(), file: S.meta.name, created: S.meta.created, view: r.view, fps: r.fps, slowmo: r.slowmo, total: sc.total, grade: sc.grade, pace_s: r.pace_s || null,
      metrics: Object.fromEntries(["cadence", "gct", "gctL", "gctR", "flight", "dutyFactor", "vo", "vr", "stepLen", "shankIC", "kneeIC", "footIC", "footStrike", "footAhead", "trunkLean", "hipExt", "kneeSwing", "kneeStance", "elbow", "armAmp", "gctAsym", "pelvicDrop", "fppa", "footLateral", "stepWidth", "armCross", "sway", "kvert", "kleg", "Fmax_bw", "nContacts", "detectRate"].map((k) => [k, r[k] ?? null])),
      coros: S.win ? { labelId: S.win.labelId, offset: S.win.offset, pace_s: S.win.pace_s, cad: S.win.cad, sl: S.win.sl, gct: S.win.gct, vo: S.win.vo, vr: S.win.vr, hr: S.win.hr, pw: S.win.pw } : null,
      scores: Object.fromEntries(sc.rows.map((x) => [x.key, x.score])), issues: F.buildFindings(r, sc).issues.map((i) => i.title), warnings: r.warnings,
      thumbs: S.thumbs.slice(0, 2).map((t) => ({ label: t.label, dataUrl: t.dataUrl })) };
    const hist = store.get(); hist.unshift(entry); store.set(hist.slice(0, 24)); renderHistory();
  }
  function renderHistory() {
    const el = q("#fm-history"); if (!el) return; const hist = store.get();
    el.innerHTML = `<div class="flex flex-wrap items-center justify-between gap-2"><div class="font-black text-lg">📚 分析歷史（此裝置）</div><div class="flex gap-2"><button class="px-3 py-1.5 rounded-lg bg-slate-200 text-xs font-bold" onclick="FORM.exportHistory()">匯出</button><label class="px-3 py-1.5 rounded-lg bg-slate-200 text-xs font-bold cursor-pointer">匯入<input type="file" accept="application/json" class="hidden" onchange="FORM.importHistory(this.files[0])"></label></div></div>
      ${hist.length ? `<div class="overflow-x-auto scrollbar-thin mt-3"><table class="w-full text-xs"><thead><tr class="text-left text-slate-500"><th class="py-1">時間</th><th>視角</th><th>配速</th><th>總分</th><th>步頻</th><th>GCT</th><th>振幅</th><th>${"小腿角／骨盆"}</th><th>主要課題</th><th></th></tr></thead><tbody>
        ${hist.map((e) => `<tr class="border-t"><td class="py-1 whitespace-nowrap">${h(new Date(e.ts).toLocaleString("zh-TW", { hour12: false }).slice(0, 16))}</td><td>${{ side: "側", front: "正", back: "背" }[e.view] || e.view}</td><td class="font-mono">${paceStr(e.pace_s)}</td><td class="font-black ${scoreColor(e.total)}">${e.total ?? "-"} ${e.grade}</td><td class="font-mono">${e.metrics.cadence ?? "-"}</td><td class="font-mono">${e.metrics.gct ?? "-"}</td><td class="font-mono">${e.metrics.vo ?? "-"}</td><td class="font-mono">${e.view === "side" ? (e.metrics.shankIC ?? "-") + "°" : (e.metrics.pelvicDrop ?? "-") + "°"}</td><td class="text-slate-600">${h((e.issues || []).slice(0, 2).join("、"))}</td><td><button class="text-rose-500" onclick="FORM.deleteHistory(${e.id})">✕</button></td></tr>`).join("")}
      </tbody></table></div>
      ${hist.length >= 2 ? `<div class="text-xs text-slate-500 mt-2">趨勢：步頻 ${trend(hist, "cadence")} · 觸地時間 ${trend(hist, "gct")} · 垂直振幅 ${trend(hist, "vo")} · 總分 ${hist[0].total ?? "-"}（上次 ${hist[1].total ?? "-"}）</div>` : ""}` : `<div class="text-sm text-slate-500 mt-2">尚無分析。上傳第一段影片後，這裡會累積每次的指標，方便追蹤改善。</div>`}`;
  }
  const trend = (hist, k) => { const a = hist[0].metrics[k], b = hist[1].metrics[k]; return isNum(a) && isNum(b) ? `${a}（${a - b >= 0 ? "+" : ""}${(a - b).toFixed(1)}）` : "-"; };
  function deleteHistory(id) { store.set(store.get().filter((e) => e.id !== id)); renderHistory(); }
  function exportHistory() { const blob = new Blob([JSON.stringify(store.get(), null, 1)], { type: "application/json" }); const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `form_history_${new Date().toISOString().slice(0, 10)}.json`; a.click(); }
  function importHistory(file) { if (!file) return; file.text().then((t) => { const arr = JSON.parse(t); if (!Array.isArray(arr)) throw new Error(); const cur = store.get(); const ids = new Set(cur.map((e) => e.id)); store.set(cur.concat(arr.filter((e) => e && e.id && !ids.has(e.id))).sort((a, b) => b.id - a.id).slice(0, 48)); renderHistory(); }).catch(() => alert("匯入失敗：不是有效的歷史檔")); }
  function downloadJson() { if (!S.res) return; const payload = { meta: S.meta, result: S.res, score: S.score, coros: S.win, generated: new Date().toISOString() }; const blob = new Blob([JSON.stringify(payload, null, 1)], { type: "application/json" }); const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `form_analysis_${new Date().toISOString().replace(/[:T]/g, "-").slice(0, 16)}.json`; a.click(); }

  function summaryText() {
    const r = S.res, sc = S.score, w = S.win; if (!r) return "";
    const fd = F.buildFindings(r, sc);
    const L = [`🎥 跑姿分析 ${new Date().toLocaleString("zh-TW", { hour12: false }).slice(0, 16)} · ${{ side: "側面", front: "正面", back: "背面" }[r.view]} · ${r.fps} fps${r.slowmo > 1 ? ` (慢動作 ${r.slowmo}×)` : ""} · 總分 ${sc.total} (${sc.grade})${r.pace_s ? ` · 配速 ${paceStr(r.pace_s)}` : ""}`];
    if (w) L.push(`COROS 同時段：${paceStr(w.pace_s)}/km, ${w.cad} spm, 步長 ${w.sl ? (w.sl / 1000).toFixed(2) : "-"} m, GCT ${w.gct} ms, VO ${w.vo != null ? (w.vo / 10).toFixed(1) : "-"} cm, VR ${w.vr}%, HR ${w.hr}, ${w.pw} W (訓練 ${w.labelId} 第 ${w.offset} 秒)`);
    L.push(`影片：步頻 ${r.cadence} · GCT ${r.gct}±${r.gctTol} ms (L ${r.gctL}/R ${r.gctR}) · 騰空 ${r.flight} ms · 觸地佔比 ${r.dutyFactor ?? "-"} · VO ${r.vo} cm${r.vr ? ` · VR ${r.vr}%` : ""}`);
    if (r.view === "side") L.push(`側面：小腿角(IC) ${r.shankIC}° · 腳在骨盆前 ${r.footAhead} cm · 膝(IC) ${r.kneeIC}° · 著地 ${r.footStrike} (足角 ${r.footIC}°) · 前傾 ${r.trunkLean}° · 髖伸展(TO) ${r.hipExt}° · 後擺膝 ${r.kneeSwing}° · 手肘 ${r.elbow}°`);
    else L.push(`${r.view === "front" ? "正面" : "背面"}：骨盆下沉 ${r.pelvicDrop}° (L ${r.perSide?.L?.pelvicDrop}/R ${r.perSide?.R?.pelvicDrop}) · FPPA ${r.fppa}° · 著地橫向 ${r.footLateral} cm (步寬 ${r.stepWidth}) · 手臂過中線 ${r.armCross}% · 側擺 ${r.sway} cm · 手肘 ${r.elbow}°`);
    if (r.kvert) L.push(`Morin 模型：Fmax ${r.Fmax_bw} BW · 重心下沉 ${r.comDrop} cm · k_vert ${r.kvert} kN/m · k_leg ${r.kleg} kN/m`);
    if (fd.issues.length) L.push("優先課題：" + fd.issues.map((i, k) => `${k + 1}. ${i.title}（${i.score}）`).join("；"));
    if (r.warnings.length) L.push("注意：" + r.warnings.join("；"));
    return L.join("\n");
  }
  function copySummary() { const t = summaryText(); const done = () => { const el = q("#fm-copied"); if (el) { el.textContent = "已複製 ✓"; setTimeout(() => (el.textContent = ""), 2500); } }; if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(t).then(done).catch(() => prompt("複製以下內容：", t)); else prompt("複製以下內容：", t); }

  function renderReviews() {
    const el = q("#fm-reviews"); if (!el) return; const D = global.D || {}; const revs = D.form_reviews || [];
    el.innerHTML = `<div class="font-black text-lg">🧑‍🏫 教練影片回顧</div><div class="text-xs text-slate-500 mt-1">把上面的「教練摘要」或影片貼到 Google Chat 教練室，AI 教練會把專業回顧與對課表的調整整理到這裡（需密碼）。</div>
      ${revs.length ? revs.map((e) => `<div class="mt-3 rounded-2xl border p-4 stripe" style="border-color:#a855f7"><div class="text-xs text-slate-500 font-bold">${h(e.ts)} ${(e.tags || []).map((t) => `<span class="badge bg-violet-100 text-violet-700">#${h(t)}</span>`).join(" ")}</div><div class="font-black mt-1">${h(e.title || e.summary || "")}</div><div class="text-sm text-slate-700 mt-1 whitespace-pre-line">${h(e.review || "")}</div>${(e.drills || []).length ? `<div class="mt-2 text-sm bg-amber-50 border border-amber-200 rounded-xl p-3"><b>📌 處方／課表調整：</b><ul class="list-disc pl-5 mt-1">${e.drills.map((d) => `<li>${h(d)}</li>`).join("")}</ul></div>` : ""}</div>`).join("") : `<div class="text-sm text-slate-500 mt-2">尚無教練回顧。</div>`}`;
  }

  global.FORM = Object.assign(F, { renderFormTab, renderResults, renderReviews, renderHistory, deleteHistory, exportHistory, importHistory, downloadJson, copySummary, summaryText, state: S });
})(typeof window !== "undefined" ? window : globalThis);
