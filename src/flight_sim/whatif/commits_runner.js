#!/usr/bin/env node
/* Runs Jarvis's quick model (engine.js, the same file the predictions page uses) on many designs.
 *
 *   node commits_runner.js < job.json > result.json
 *
 * job.json:  { runs: [ { id, base, ref, conds: { sim: {railAngleDeg, rodLength, windMs, padK, padPa, padAlt} } } ] }
 * result:    { id: { sim: { apogee, max_mach, ... } } }, the History table's metric names and units
 * (metres, kPa, calibers and % of body length), so the History page can draw them as they are.
 * Only the climb is flown: the way down is not needed here.
 */
'use strict';
const fs = require('fs');
const W = require('./engine.js');

const round = (v, d) => (Number.isFinite(v) ? Number(v.toFixed(d)) : null);

function design(base) {   // the design as the page builds it from the base data
  return {
    finCount: base.fins.count, finRoot: base.fins.root, finTip: base.fins.tip, finSpan: base.fins.span,
    finSweep: base.fins.sweep, finThick: base.fins.thick, noseLength: base.noseLength, bodyLength: base.bodyLength,
    tailLength: base.tailLength, tailAft: base.tailAftRadius,
  };
}

function runOne(run) {
  const base = run.base, change = design(base);
  base.geometry = W.applyChange(base, change);
  base.refGeometry = W.applyChange(base, run.ref || change);
  const out = {};
  for (const [sim, cond] of Object.entries(run.conds)) {
    const r = W.fly(base, change, Object.assign({}, cond, { leanIntoWind: true }), { descent: false });
    const s = r.sum, pct = (cal) => cal * base.diameter / r.geo.length * 100;
    out[sim] = {
      apogee: round(s.apogee, 1),
      max_mach: round(s.machMax, 4),
      max_dynamic_pressure_kpa: round(s.qMax / 1000, 4),
      stability_off_rod_cal: round(s.marginRail, 4),
      min_stability_cal: round(s.marginLo, 4),
      max_stability_cal: round(s.marginHi, 4),
      stability_off_rod_pct: round(pct(s.marginRail), 3),
      min_stability_pct: round(pct(s.marginLo), 3),
      max_stability_pct: round(pct(s.marginHi), 3),
    };
  }
  return out;
}

const job = JSON.parse(fs.readFileSync(0, 'utf8'));
const result = {};
for (const run of job.runs) {
  try { result[run.id] = runOne(run); } catch (e) { result[run.id] = { error: String(e.message || e) }; }
}
process.stdout.write(JSON.stringify(result));
