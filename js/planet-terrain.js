import { MOON_DEM, MOON_DEM_GZIP } from './moon-data.js';

// Local equirectangular tangent approximation: east is +x, south is +z.
export const MOON_METRES_PER_DEGREE = MOON_DEM.radius * Math.PI / 180;
export const MOON_SPAN_Z = (MOON_DEM.north - MOON_DEM.south) * MOON_METRES_PER_DEGREE;
export const MOON_SPAN_X = (MOON_DEM.east - MOON_DEM.west) * MOON_METRES_PER_DEGREE * Math.cos(MOON_DEM.centerLatitude * Math.PI / 180);

export async function decodeMoonTerrain() {
  const packed = Uint8Array.from(atob(MOON_DEM_GZIP), (c) => c.charCodeAt(0));
  const stream = new Blob([packed]).stream().pipeThrough(new DecompressionStream('gzip'));
  const raw = await new Response(stream).arrayBuffer();
  if (raw.byteLength !== MOON_DEM.width * MOON_DEM.height * 2) throw new Error('Invalid Moon elevation asset');
  const source = new DataView(raw);
  const heights = new Float32Array(MOON_DEM.width * MOON_DEM.height);
  for (let i = 0; i < heights.length; i++) heights[i] = source.getInt16(i * 2, true);
  return heights;
}

// Only the height bake and flight probe pipelines bind the source DEM.
export const PLANET_HEIGHT_WGSL = /* wgsl */ `
@group(0) @binding(3) var<storage, read> planetHeights: array<f32>;
// Cubic B-spline reconstruction: positive weights sum to one, so the surface
// stays inside the neighbourhood's elevation range. Unlike bilinear sampling,
// its first and second derivatives remain continuous at source-cell boundaries.
fn planetSplineWeights(t: f32) -> vec4f {
  let t2 = t * t;
  let t3 = t2 * t;
  let s = 1.0 - t;
  return vec4f(s * s * s, 3.0 * t3 - 6.0 * t2 + 4.0,
    -3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0, t3) / 6.0;
}
fn planetHeight(xz: vec2f) -> f32 {
  let size = vec2i(${MOON_DEM.width}, ${MOON_DEM.height});
  let uv = xz / vec2f(${MOON_SPAN_X}, ${MOON_SPAN_Z}) + 0.5;
  let p = clamp(uv * vec2f(size) - 0.5, vec2f(0.0), vec2f(size - 1));
  let a = vec2i(floor(p));
  let wx = planetSplineWeights(fract(p.x));
  let wz = planetSplineWeights(fract(p.y));
  var height = 0.0;
  for (var z = 0; z < 4; z++) {
    let row = clamp(a.y + z - 1, 0, size.y - 1) * size.x;
    var line = 0.0;
    for (var x = 0; x < 4; x++) {
      let col = clamp(a.x + x - 1, 0, size.x - 1);
      line += planetHeights[row + col] * wx[x];
    }
    height += line * wz[z];
  }
  return height;
}
`;
