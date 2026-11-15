// Northbound regional routes, in local metres. Quintic easing keeps position,
// heading and acceleration continuous without overshooting waypoint bounds.
export function createCameraRoute(waypoints) {
  if (waypoints.length < 2) throw new Error('A camera route needs at least two waypoints');
  const points = waypoints.map((p) => ({ ...p }));
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    if (![p.x, p.z, p.agl, p.pitch].every(Number.isFinite) || p.agl <= 0
        || (i > 0 && p.z >= points[i - 1].z)) throw new Error('Invalid northbound camera route');
  }
  function segment(z) {
    for (let i = 0; i < points.length - 2; i++) if (z >= points[i + 1].z) return i;
    return points.length - 2;
  }
  function value(z, key) {
    const i = segment(z), a = points[i], b = points[i + 1];
    const t = Math.max(0, Math.min(1, (a.z - z) / (a.z - b.z)));
    const blend = t * t * t * (t * (t * 6 - 15) + 10);
    return a[key] + (b[key] - a[key]) * blend;
  }
  function slope(z) {
    const i = segment(z), a = points[i], b = points[i + 1];
    const t = Math.max(0, Math.min(1, (a.z - z) / (a.z - b.z)));
    return -(b.x - a.x) * 30 * t * t * (t - 1) * (t - 1) / (a.z - b.z);
  }
  const startZ = points[0].z, endZ = points.at(-1).z;
  let length = 0, previousX = value(startZ, 'x'), previousZ = startZ;
  for (let z = startZ - 100; previousZ > endZ; z -= 100) {
    const nextZ = Math.max(endZ, z), nextX = value(nextZ, 'x');
    length += Math.hypot(nextX - previousX, nextZ - previousZ);
    previousX = nextX; previousZ = nextZ;
  }
  return { startZ, endZ, length, slope,
    x: (z) => value(z, 'x'), agl: (z) => value(z, 'agl'), pitch: (z) => value(z, 'pitch') };
}
