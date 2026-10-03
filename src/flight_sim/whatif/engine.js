/* Flight model for the predictions page: a planar 3-DOF sim (up, downrange, pitch) driven by the team's RASAero table.
 *
 * The RASAero table is for the geometry RASAero was given (base.refGeometry, which is the design itself
 * when no RASAero file is named). Any other geometry, a hypothetical one or a newer .ork, is flown by
 * correcting that table:
 *   normal force   scaled by the ratio of Barrowman CN-alpha, new / RASAero's
 *   centre of pressure   shifted by the Barrowman CP shift
 *   axial force    plus the change in fin, body-friction, nose-wave and base drag from RASAero's
 * Mass, CG and pitch inertia come from the component masses, with the stretch and added shell mass
 * of the change. Nothing here is OpenRocket or RASAero: it is a quick estimate, checked against the
 * 6-DOF sim and OpenRocket on the design as committed (see the page's "model check").
 * Units are SI throughout; x is measured from the nose tip, positive aft.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.WhatIf = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';
  const G0 = 9.80665, RAIR = 287.053, GAMMA = 1.4, REARTH = 6371000, LAPSE = 0.0065;
  const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

  /* ---------- atmosphere: troposphere from the pad temperature, isothermal above ---------- */
  function atmosphere(padK, padPa, h) {
    const hTrop = 11000, tTop = padK - LAPSE * hTrop;
    let T, p;
    if (h <= hTrop) {
      T = padK - LAPSE * h;
      p = padPa * Math.pow(T / padK, G0 / (RAIR * LAPSE));
    } else {
      const pTop = padPa * Math.pow(tTop / padK, G0 / (RAIR * LAPSE));
      T = tTop;
      p = pTop * Math.exp(-G0 * (h - hTrop) / (RAIR * tTop));
    }
    const rho = p / (RAIR * T);
    const mu = 1.458e-6 * Math.pow(T, 1.5) / (T + 110.4);
    return { T, p, rho, a: Math.sqrt(GAMMA * RAIR * T), mu };
  }

  /* ---------- Barrowman: CN-alpha and centre of pressure of the geometry ---------- */
  function barrowman(g) {
    const d = g.diameter, area = Math.PI * d * d / 4, R = d / 2;
    const parts = [];
    // nose
    const xN = ({ haack: 0.437, ogive: 0.466, conical: 0.667, ellipsoid: 0.333, parabolic: 0.5, power: 0.5 }[g.noseShape] || 0.437) * g.noseLength;
    parts.push({ name: 'nose', cna: 2, x: xN });
    // transition (tail)
    const dF = 2 * g.tailForeRadius, dA = 2 * g.tailAftRadius;
    if (Math.abs(dF - dA) > 1e-9) {
      const cna = 2 * ((dA / d) ** 2 - (dF / d) ** 2);
      const r = dF / dA;
      const xT = g.tailX + g.tailLength / 3 * (1 + (1 - r) / (1 - r * r));
      parts.push({ name: 'tail', cna, x: xT });
    }
    // fins (Barrowman, with body-fin interference)
    const f = g.fins, s = f.span;
    const lm = Math.hypot(s, f.sweep + (f.tip - f.root) / 2);
    const kfb = 1 + R / (s + R);
    const cnaF = kfb * (4 * f.count * (s / d) ** 2) / (1 + Math.sqrt(1 + (2 * lm / (f.root + f.tip)) ** 2));
    const xf = f.leX + (f.sweep * (f.root + 2 * f.tip)) / (3 * (f.root + f.tip)) +
      ((f.root + f.tip) - (f.root * f.tip) / (f.root + f.tip)) / 6;
    parts.push({ name: 'fins', cna: cnaF, x: xf });
    const cna = parts.reduce((a, p) => a + p.cna, 0);
    const xcp = parts.reduce((a, p) => a + p.cna * p.x, 0) / cna;
    return { parts, cna, xcp, area };
  }

  /* ---------- geometry from the baseline plus the sliders ---------- */
  function applyChange(base, c) {
    // base: parsed design; c: {finCount, finRoot, finTip, finSpan, finSweep, finThick, noseLength, bodyLength, tailLength, tailAft}
    const dn = c.noseLength - base.noseLength, dl = c.bodyLength - base.bodyLength;
    const bodyEnd0 = base.bodyStart + base.bodyLength;
    const bodyEnd1 = bodyEnd0 + dn + dl;
    const g = {
      diameter: base.diameter, noseShape: base.noseShape, noseLength: c.noseLength,
      tailLength: c.tailLength, tailForeRadius: base.tailForeRadius, tailAftRadius: c.tailAft,
      tailX: bodyEnd1, length: bodyEnd1 + c.tailLength, bodyStart: base.bodyStart + dn, bodyLength: c.bodyLength,
      bodyEnd: bodyEnd1,
      fins: {
        count: c.finCount, root: c.finRoot, tip: c.finTip, span: c.finSpan, sweep: c.finSweep, thick: c.finThick,
        teX: bodyEnd1 - (bodyEnd0 - base.fins.teX),
      },
    };
    g.fins.leX = g.fins.teX - c.finRoot;
    // moves each mass station with the stretch
    const move = (x) => {
      if (x < base.noseLength) return x * 1;                    // inside the nose: stays
      let y = x + dn;
      if (x > base.bodyStart + base.bodyLength) y += dl;
      else if (x > base.bodyStart) y += dl * (x - base.bodyStart) / base.bodyLength;
      return y;
    };
    // mass lumps: [mass, x, length]
    const lumps = [];
    const rho = base.fins.density;
    for (const it of base.items) {
      let m = it.mass, x = move(it.x), len = it.len;
      if (it.kind === 'fins') continue;
      if (it.kind === 'nosecone') { m = it.mass * c.noseLength / base.noseLength; x = 0.5 * c.noseLength; len = 0; }
      if (it.kind === 'transition') {
        const a0 = shellArea(base.tailLength, base.tailForeRadius, base.tailAftRadius);
        const a1 = shellArea(c.tailLength, base.tailForeRadius, c.tailAft);
        m = it.mass + base.tailWall * base.tailDensity * (a1 - a0);
        x = bodyEnd1 + 0.5 * c.tailLength;
      }
      lumps.push([m, x, len]);
    }
    // fins: plate of the fin material, centroid of the trapezoid
    const fa = 0.5 * (c.finRoot + c.finTip) * c.finSpan;
    const fm = c.finCount * fa * c.finThick * rho;
    const fx = g.fins.leX + (c.finSweep * (c.finRoot + 2 * c.finTip)) / (3 * (c.finRoot + c.finTip)) +
      (c.finRoot ** 2 + c.finRoot * c.finTip + c.finTip ** 2) / (3 * (c.finRoot + c.finTip));
    lumps.push([fm, fx, 0]);
    // added body shell for a longer (or shorter) combustion chamber tube
    if (Math.abs(dl) > 1e-9) {
      const lin = base.bodyDensity * Math.PI * (base.diameter ** 2 / 4 - (base.diameter / 2 - base.bodyWall) ** 2);
      lumps.push([lin * dl, g.bodyStart + 0.5 * c.bodyLength, 0]);
    }
    // motor: mass fixed, at the middle of the stretched chamber
    const motor = { mass: base.motor.mass, prop: base.motor.prop, x: g.bodyStart + 0.5 * c.bodyLength, len: c.bodyLength };
    g.lumps = lumps; g.motor = motor;
    g.finMass = fm; g.finArea = fa;
    return g;
  }
  function shellArea(L, r1, r2) { return Math.PI * (r1 + r2) * Math.hypot(L, r1 - r2); }

  function massAt(g, burned) {
    const L = g.lumps.concat([[g.motor.mass - burned * g.motor.prop, g.motor.x, g.motor.len]]);
    let m = 0, mx = 0;
    for (const [mi, xi] of L) { m += mi; mx += mi * xi; }
    const cg = mx / m;
    let iyy = 0;
    for (const [mi, xi, li] of L) iyy += mi * ((xi - cg) ** 2 + li * li / 12);
    return { m, cg, iyy };
  }

  /* ---------- skin friction and the drag changes ---------- */
  function cf(Re, M) {
    const re = Math.max(Re, 1e6);
    return 0.455 / Math.pow(Math.log10(re), 2.58) * Math.pow(1 + 0.144 * M * M, -0.65);
  }
  // drag of the parts a geometry change touches, as a coefficient on the reference area
  function partDrag(g, M, atm, V) {
    const area = Math.PI * g.diameter ** 2 / 4;
    const f = g.fins, c = 0.5 * (f.root + f.tip), Sf = 0.5 * (f.root + f.tip) * f.span;
    const Re = (len) => atm.rho * V * len / atm.mu;
    // fins: friction both sides, thickness factor, and wave drag on the swept leading edge
    const tc = f.thick / c;
    const dfr = f.count * 2 * Sf * cf(Re(c), M) * (1 + 2 * tc) / area;
    const lam = Math.atan2(f.sweep, f.span);                     // leading-edge sweep angle
    const mn = M * Math.cos(lam);
    const k = Math.pow(clamp((mn - 0.6) / 0.6, 0, 1), 2);
    const dw = f.count * Sf * k * 4 * tc * tc / Math.sqrt(Math.max(mn, 1.2) ** 2 - 1) / area;
    // body: friction on the cylinder length and the nose surface
    const cyl = Math.PI * g.diameter * g.bodyLength * 1.0;
    const noseWet = 0.9 * Math.PI * g.diameter * g.noseLength;
    const body = (cyl + noseWet) * cf(Re(g.length), M) / area;
    // nose wave drag, a slender-body estimate that fades below Mach 0.9
    const lamN = g.noseLength / g.diameter;
    const nw = 0.9 * Math.pow(1 / (2 * lamN), 2) * clamp((M - 0.9) / 0.4, 0, 1);
    // base drag of the boat tail: OpenRocket's base-pressure form on the aft area
    const cb = M < 1 ? 0.12 + 0.13 * M * M : 0.25 / M;
    const baseR = g.tailAftRadius;
    const base = cb * (baseR * baseR) / (g.diameter * g.diameter / 4);
    return dfr + dw + body + nw + base;
  }

  /* ---------- the aero table, with the corrections ---------- */
  function makeAero(table, base, geo) {
    const ref = base.refGeometry || base.geometry;
    const b0 = barrowman(ref), b1 = barrowman(geo);
    const cnScale = b1.cna / b0.cna, shift = b1.xcp - b0.xcp;
    const M = table.mach, A = table.alpha;
    const idx = (arr, v) => {
      let lo = 0, hi = arr.length - 1;
      if (v <= arr[0]) return [0, 0, 0];
      if (v >= arr[hi]) return [hi, hi, 0];
      while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (arr[mid] <= v) lo = mid; else hi = mid; }
      return [lo, hi, (v - arr[lo]) / (arr[hi] - arr[lo])];
    };
    const bil = (grid, m, a) => {
      const [m0, m1, tm] = idx(M, m), [a0, a1, ta] = idx(A, a);
      const lo = grid[m0][a0] * (1 - ta) + grid[m0][a1] * ta, hi = grid[m1][a0] * (1 - ta) + grid[m1][a1] * ta;
      return lo * (1 - tm) + hi * tm;
    };
    const same = Math.abs(cnScale - 1) < 1e-12 && Math.abs(shift) < 1e-12;
    return {
      cnScale, shift, b0, b1,
      ca: (m, aDeg, atm, V) => bil(table.ca, m, aDeg) + partDrag(geo, m, atm, V) - partDrag(ref, m, atm, V),
      cn: (m, aDeg) => bil(table.cn, m, aDeg) * cnScale,
      xcp: (m, aDeg) => bil(table.xcp, m, Math.max(aDeg, 1)) + shift,
      same,
    };
  }

  /* ---------- the flight ---------- */
  function fly(base, change, cond, opt) {
    opt = opt || {};
    const geo = applyChange(base, change);
    const aero = makeAero(base.aero, base, geo);
    const area = Math.PI * geo.diameter ** 2 / 4, D = geo.diameter;
    const thrust = base.motor.thrust;                           // [[t, N], ...]
    const thr = (t) => {
      if (t <= thrust[0][0]) return thrust[0][1] * Math.max(0, t / Math.max(thrust[0][0], 1e-9));
      for (let i = 1; i < thrust.length; i++) if (t <= thrust[i][0]) {
        const [t0, f0] = thrust[i - 1], [t1, f1] = thrust[i];
        return f0 + (f1 - f0) * (t - t0) / (t1 - t0);
      }
      return 0;
    };
    const burnEnd = thrust[thrust.length - 1][0];
    // pitch-damping sum of Barrowman CNa * (x - cg)^2, per rad
    const dampParts = aero.b1.parts;
    const tilt0 = cond.railAngleDeg * Math.PI / 180;
    const sgnW = cond.leanIntoWind === false ? -1 : 1;          // wind blows toward -y when leaning into it
    const wy = -sgnW * cond.windMs;
    const rodLen = cond.rodLength;
    const padK = cond.padK, padPa = cond.padPa;
    let t = 0, x = 0, y = 0, vx = 0, vy = 0, th = tilt0, om = 0;
    let onRail = true, railS = 0;
    const out = { t: [], alt: [], y: [], v: [], vup: [], mach: [], acc: [], aoa: [], marginCal: [], q: [], cd: [], mass: [] };
    const sum = { apogee: 0, apogeeT: 0, vmax: 0, machMax: 0, accMax: 0, qMax: 0, railV: NaN, railT: NaN, marginRail: NaN, marginMin: Infinity,
      burnoutAlt: NaN, burnoutV: NaN, drift: 0, aoaMax: 0, marginBurnout: NaN };
    const dt = opt.dt || 0.01, rec = opt.record === false ? 0 : (opt.every || 0.1);
    let nextRec = 0, ax = 0, ay = 0;
    const deriv = (s, tt) => {
      const [X, Y, VX, VY, TH, OM] = s;
      const burned = clamp(tt / burnEnd, 0, 1);
      const mp = massAt(geo, burned);
      const atm = atmosphere(padK, padPa, Math.max(X, 0));
      const g = G0 * (REARTH / (REARTH + X + cond.padAlt)) ** 2;
      const rvx = VX, rvy = VY - wy;
      const V = Math.hypot(rvx, rvy);
      const nx = Math.cos(TH), ny = Math.sin(TH), ex = -Math.sin(TH), ey = Math.cos(TH);
      const Vax = rvx * nx + rvy * ny, Vc = rvx * ex + rvy * ey;
      const aoa = Math.atan2(Math.abs(Vc), Math.abs(Vax));
      const aDeg = clamp(aoa * 180 / Math.PI, 0, 30);
      const M = V / atm.a, q = 0.5 * atm.rho * V * V;
      const sAx = Vax >= 0 ? 1 : -1;
      const CA = aero.ca(M, aDeg, atm, Math.max(V, 1)), CN = aero.cn(M, aDeg);
      const Fax = -sAx * CA * q * area * Math.cos(aoa) ;       // along the body axis
      const sC = Vc >= 0 ? 1 : -1;
      const Fn = -sC * CN * q * area;                           // along e
      const T = thr(tt);
      const fx = (T + Fax) * nx + Fn * ex - mp.m * g, fy = (T + Fax) * ny + Fn * ey;
      const arm = aero.xcp(M, aDeg) - mp.cg;
      let tau = sC * CN * q * area * arm;                       // restoring when the CP is aft of the CG
      // pitch damping
      let cq = 0; for (const p of dampParts) cq += p.cna * (p.x - mp.cg) ** 2;
      tau += -0.5 * atm.rho * V * area * cq * OM;
      if (onRail) return { d: [VX, VY, 0, 0, 0, 0], on: true, T, Fax, Fn, mp, atm, g, M, q, V, aoa, CA, CN, arm, fx, fy, nx, ny };
      return { d: [VX, VY, fx / mp.m, fy / mp.m, OM, tau / mp.iyy], T, Fax, Fn, mp, atm, g, M, q, V, aoa, CA, CN, arm, fx, fy, nx, ny, on: false };
    };
    let guard = 0;
    while (t < 400 && guard++ < 200000) {
      const s0 = [x, y, vx, vy, th, om];
      let r1;
      if (onRail) {
        // along the rail: thrust, drag and gravity component along the axis
        const nx = Math.cos(tilt0), ny = Math.sin(tilt0);
        const mp = massAt(geo, clamp(t / burnEnd, 0, 1));
        const sRail = railS, vr = vx * nx + vy * ny;
        const atm = atmosphere(padK, padPa, Math.max(x, 0));
        const V = Math.hypot(vx, vy - wy), Vax = vr, M = V / atm.a;
        const rvx = vx, rvy = vy - wy;
        const q = 0.5 * atm.rho * V * V;
        const CA = aero.ca(M, 0, atm, Math.max(V, 1));
        const T = thr(t);
        let a = (T - CA * q * area * (Vax >= 0 ? 1 : -1) - mp.m * G0 * nx) / mp.m;
        if (T < mp.m * G0 * nx && railS <= 0 && vr <= 0) a = 0;       // on the pad until it lifts
        // semi-implicit step
        const vr1 = Math.max(vr + a * dt, 0);
        railS = sRail + 0.5 * (vr + vr1) * dt;
        vx = vr1 * nx; vy = vr1 * ny; x = railS * nx; y = railS * ny;
        ax = a * nx; ay = a * ny;
        t += dt;
        if (railS >= rodLen) { onRail = false; sum.railV = vr1; sum.railT = t;
          const e0 = deriv([x, y, vx, vy, th, om], t); sum.marginRail = (aero.xcp(e0.M, 2) - e0.mp.cg) / D; }
        continue;
      }
      // RK4
      const k1 = deriv(s0, t).d;
      const s1 = s0.map((v, i) => v + 0.5 * dt * k1[i]), k2 = deriv(s1, t + 0.5 * dt).d;
      const s2 = s0.map((v, i) => v + 0.5 * dt * k2[i]), k3 = deriv(s2, t + 0.5 * dt).d;
      const s3 = s0.map((v, i) => v + dt * k3[i]), k4 = deriv(s3, t + dt).d;
      const sN = s0.map((v, i) => v + dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
      const e = deriv(s0, t);
      ax = e.d[2]; ay = e.d[3];
      [x, y, vx, vy, th, om] = sN;
      t += dt;
      const aKin = Math.hypot(ax, ay + 0) / G0;
      const mrg = (aero.xcp(e.M, 2) - e.mp.cg) / D;
      if (e.V > sum.vmax) sum.vmax = e.V;
      if (e.M > sum.machMax) sum.machMax = e.M;
      if (aKin > sum.accMax && t > 0.2) sum.accMax = aKin;
      if (e.q > sum.qMax) sum.qMax = e.q;
      if (mrg < sum.marginMin) sum.marginMin = mrg;
      if (e.aoa * 57.2958 > sum.aoaMax && e.V > 30) sum.aoaMax = e.aoa * 57.2958;
      if (isNaN(sum.burnoutAlt) && t >= burnEnd) { sum.burnoutAlt = x; sum.burnoutV = Math.hypot(vx, vy); sum.marginBurnout = mrg; }
      if (rec && t >= nextRec) {
        nextRec += rec;
        out.t.push(t); out.alt.push(x); out.y.push(y); out.v.push(Math.hypot(vx, vy)); out.vup.push(vx);
        out.mach.push(e.M); out.acc.push(aKin); out.aoa.push(e.aoa * 57.2958); out.marginCal.push(mrg);
        out.q.push(e.q); out.cd.push(e.CA); out.mass.push(e.mp.m);
      }
      if (x > sum.apogee) { sum.apogee = x; sum.apogeeT = t; sum.drift = y; }
      if (vx < 0 && t > 1) break;
    }
    sum.length = geo.length;
    sum.mass0 = massAt(geo, 0).m; sum.mass1 = massAt(geo, 1).m;
    sum.cg0 = massAt(geo, 0).cg; sum.cg1 = massAt(geo, 1).cg;
    sum.cpBarrowman = aero.b1.xcp; sum.cnaBarrowman = aero.b1.cna;
    sum.cnScale = aero.cnScale; sum.cpShift = aero.shift;
    sum.finMass = geo.finMass;
    return { out, sum, geo, aero };
  }

  /* ---------- the CD-vs-Mach curve of a geometry, for the chart ---------- */
  function dragCurve(base, change, cond, aDeg) {
    const geo = applyChange(base, change), aero = makeAero(base.aero, base, geo);
    const res = [];
    for (let m = 0.1; m <= 3.0; m += 0.05) {
      const atm = atmosphere(cond.padK, cond.padPa, 2500);
      res.push([m, aero.ca(m, aDeg || 0, atm, m * atm.a)]);
    }
    return res;
  }

  return { fly, applyChange, barrowman, atmosphere, massAt, dragCurve, makeAero, partDrag };
});
