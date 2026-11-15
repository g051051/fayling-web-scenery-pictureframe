// Instanced trees. A compute pass walks the jittered tree grid around the camera, keeps the
// trees that can be on screen and appends them to a near list (full crown) and a far list
// (cheap crown) that two indirect draws consume. Every tree is real geometry at every distance,
// so flying forward never grows or inserts a tree; a crown that shrinks below a pixel is only
// recoloured towards the stand colour the ground already carries.
//
// Near crowns are not cones. A fir is a pointed skirt with a shallow branch notch and a
// ragged outline. A broadleaf is the same cluster of leaf clumps at every distance, so the
// crown never changes shape as the camera approaches. Juniper is a low cushion. The far
// list (firs past the near distance) stays a cone and fades to the stand.
import { FRAME_WGSL, ATMO_CORE_WGSL, NOISE_WGSL, AP_RES, AP_MAX_KM,
  TREE_RANGE, TREE_NEAR_DIST, TREE_MAX, TREE_NEAR_MAX, TREE_CLUMP_DETAIL, TREE_CLUMP_DETAIL_MAX, TREE_CLUMP_COARSE_MAX } from './common.js';
import { SKY_RENDER_FUNCS } from './sky.js';
import { TERRAIN_FUNCS, clipSampleWGSL } from './terrain.js';
import { FOREST_FUNCS } from './forest.js';
import { RAIN_STRUCT } from './rain.js';

export const TREE_INST_BYTES = 80;
export const TREE_GEN_GROUP = 8;
export const TREE_SEGS_NEAR = 12;
export const TREE_RINGS_NEAR = 6;
export const TREE_TRUNK_SIDES = 6;
export const TREE_SEGS_FAR = 6;
// near: four bands of the crown shell, plus a six-sided stem
const TREE_CROWN_TRIS_NEAR = (TREE_RINGS_NEAR - 1) * TREE_SEGS_NEAR * 2;
const TREE_TRUNK_TRIS_NEAR = TREE_TRUNK_SIDES * 2;
export const TREE_VERTS_NEAR = (TREE_CROWN_TRIS_NEAR + TREE_TRUNK_TRIS_NEAR) * 3;
// clumps: a broadleaf crown is twelve overlapping lumps (an apex, a ring of three, a ring
// of five and a skirt of three), each an eight-sided ellipsoid in three bands, plus the stem
export const TREE_CLUMPS = 12;
export const TREE_LUMP_SEGS = 8;
const TREE_LUMP_TRIS = TREE_LUMP_SEGS * 4;
const TREE_CLUMP_CROWN_TRIS = TREE_CLUMPS * TREE_LUMP_TRIS;
export const TREE_VERTS_CLUMP = (TREE_CLUMP_CROWN_TRIS + TREE_TRUNK_TRIS_NEAR) * 3;
// the same twelve clumps, each an octahedron, for crowns where a clump is only a few pixels
export const TREE_COARSE_LUMP_TRIS = 8;
export const TREE_VERTS_COARSE = (TREE_CLUMPS * TREE_COARSE_LUMP_TRIS + TREE_TRUNK_TRIS_NEAR) * 3;
// far: a cone fan plus a camera-facing stem quad
export const TREE_VERTS_FAR = (TREE_SEGS_FAR + 2) * 3;

const TREE_CONSTS = /* wgsl */ `
const TREE_RANGE: f32 = ${TREE_RANGE}.0;
const TREE_NEAR_DIST: f32 = ${TREE_NEAR_DIST}.0;
const TREE_MAX: u32 = ${TREE_MAX}u;
const TREE_NEAR_MAX: u32 = ${TREE_NEAR_MAX}u;
const EARTH_R: f32 = 6371000.0;
const TREE_SEGS_NEAR: u32 = ${TREE_SEGS_NEAR}u;
const TREE_RINGS_NEAR: u32 = ${TREE_RINGS_NEAR}u;
const TREE_TRUNK_SIDES: u32 = ${TREE_TRUNK_SIDES}u;
const TREE_SEGS_FAR: u32 = ${TREE_SEGS_FAR}u;
const TREE_CLUMP_DETAIL: f32 = ${TREE_CLUMP_DETAIL}.0;
const TREE_DETAIL_MAX: u32 = ${TREE_CLUMP_DETAIL_MAX}u;
const TREE_COARSE_MAX: u32 = ${TREE_CLUMP_COARSE_MAX}u;
const TREE_FAR_MAX: u32 = ${TREE_MAX - TREE_NEAR_MAX - TREE_CLUMP_DETAIL_MAX - TREE_CLUMP_COARSE_MAX}u;
const TREE_CLUMPS: u32 = ${TREE_CLUMPS}u;
const LUMP_SEGS: u32 = ${TREE_LUMP_SEGS}u;
const LUMP_TRIS: u32 = ${TREE_LUMP_TRIS}u;
const CLUMP_CROWN_TRIS: u32 = ${TREE_CLUMP_CROWN_TRIS}u;
const COARSE_LUMP_TRIS: u32 = ${TREE_COARSE_LUMP_TRIS}u;
const COARSE_CROWN_TRIS: u32 = ${TREE_CLUMPS * TREE_COARSE_LUMP_TRIS}u;
const NEAR_CROWN_TRIS: u32 = ${TREE_CROWN_TRIS_NEAR}u;
const NEAR_TRUNK_TRIS: u32 = ${TREE_TRUNK_TRIS_NEAR}u;
struct TreeInst {
  a: vec4f,   // xyz = ground position of the trunk, w = crown radius
  b: vec4f,   // x = apex height, y = lowest crown edge, z = texture seed, w = 1 for conifers
  c: vec4f,   // x = sun visibility, y = cloud shade * sun up, z = stand density, w = conifer share
  d: vec4f,   // lit stand colour (radiance / pi), what a sub-pixel crown shows
  e: vec4f,   // this tree's foliage albedo
};
struct Light { sun: vec3f, moon: vec3f, zen: vec3f, hor: vec3f, L: vec3f, M: vec3f };
@group(0) @binding(10) var<storage, read> LB: array<vec4f, 8>;
fn readLight() -> Light {
  var lt: Light;
  lt.sun = LB[0].xyz; lt.moon = LB[1].xyz; lt.zen = LB[2].xyz;
  lt.hor = LB[3].xyz; lt.L = LB[4].xyz; lt.M = LB[5].xyz;
  return lt;
}
fn ggx(nh: f32, a: f32) -> f32 {
  let a2 = a * a;
  let d = nh * nh * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d);
}
fn ambient(lt: Light, n: vec3f) -> vec3f {
  return mix(lt.hor, lt.zen, 0.5 + 0.5 * n.y) * (0.55 + 0.45 * n.y) * PI;
}
fn direct(lt: Light, n: vec3f, sunVis: f32) -> vec3f {
  return lt.sun * sunVis * max(dot(n, lt.L), 0.0) + lt.moon * max(dot(n, lt.M), 0.0);
}
`;

// One thread: the light every tree shares this frame (sun and moon illuminance, sky colour).
export function treeLightWGSL() {
  return FRAME_WGSL + ATMO_CORE_WGSL + /* wgsl */ `
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var transLUT: texture_2d<f32>;
@group(0) @binding(2) var skyLUT: texture_2d<f32>;
@group(0) @binding(3) var sampC: sampler;
` + RAIN_STRUCT + /* wgsl */ `
@group(0) @binding(4) var<uniform> Rain: RainU;
@group(0) @binding(5) var<storage, read_write> outL: array<vec4f, 8>;
` + SKY_RENDER_FUNCS + /* wgsl */ `
fn moonIlluminance() -> vec3f {
  if (F.moonDir.w <= 0.0 || F.moonDir.y <= 0.0 || F.sunDir.y >= 0.2) { return vec3f(0.0); }
  let moon = normalize(F.moonDir.xyz);
  let lite = normalize(F.moonLight.xyz);
  let deg = acos(clamp(dot(lite, -moon), -1.0, 1.0)) * (180.0 / PI);
  let deg2 = deg * deg;
  let phase = exp(-0.921034037 * (0.026 * deg + 4e-9 * deg2 * deg2));
  let night = 1.0 - smoothstep(-0.05, 0.18, F.sunDir.y);
  let size = max(F.moonLight.w, 0.0) / 2.5;
  let E = F.moonDir.w * phase * night * 0.028 * size * size;
  return vec3f(1.0, 0.96, 0.9) * sampleTrans(camR(), moon.y) * E;
}
@compute @workgroup_size(1)
fn main() {
  let L = F.sunDir.xyz;
  let sunScale = mix(1.0, Rain.atmo.y, step(0.001, Rain.wind.w));
  let sun = sampleTrans(RG + F.lodP.z * 0.001, L.y) * F.sunDir.w * sunScale;
  var zen = skyAtmo(vec3f(0.0, 1.0, 0.0));
  var hor = (skyAtmo(normalize(vec3f(1.0, 0.25, 0.0))) + skyAtmo(normalize(vec3f(-1.0, 0.25, 0.0)))
          + skyAtmo(normalize(vec3f(0.0, 0.25, 1.0))) + skyAtmo(normalize(vec3f(0.0, 0.25, -1.0)))) * 0.25;
  let cover = clamp(F.cloudP.y * 0.8 + F.atmo.z * 0.3, 0.0, 1.0);
  let skyDim = 1.0 - 0.35 * cover;
  let glow = vec3f(0.002, 0.003, 0.005) * F.nightP.w;
  zen = zen * skyDim * sunScale + glow;
  hor = hor * skyDim * sunScale + glow;
  let nl = nightLandscape();
  zen += nl.zen;
  hor += nl.hor;
  outL[0] = vec4f(sun, 0.0);
  outL[1] = vec4f(moonIlluminance() + nl.key, 0.0);
  outL[2] = vec4f(zen, 0.0);
  outL[3] = vec4f(hor, 0.0);
  outL[4] = vec4f(L, 0.0);
  outL[5] = vec4f(select(normalize(F.moonDir.xyz), nl.dir, nl.on > 0.5), 0.0);
}
`;
}

// Walks the cells in a square around the camera. Writes the trees that survive the range,
// water, frustum and stand tests, with their lighting already folded in.
export function treeGenWGSL(filterA) {
  return FRAME_WGSL + ATMO_CORE_WGSL + /* wgsl */ `
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var sampC: sampler;
@group(0) @binding(2) var clipA: texture_2d_array<f32>;
@group(0) @binding(3) var clipB: texture_2d_array<f32>;
@group(0) @binding(4) var<uniform> CL: ClipLevels;
struct ShadowMap { origin: vec2f, texel: f32, span: f32 };
@group(0) @binding(5) var cloudShadowTex: texture_2d<f32>;
@group(0) @binding(6) var<uniform> SH: ShadowMap;
struct GenU { cell0: vec2i, dim: u32, pad: u32 };
@group(0) @binding(7) var<uniform> G: GenU;
@group(0) @binding(8) var<storage, read_write> trees: array<TreeInst>;
@group(0) @binding(9) var<storage, read_write> args: array<atomic<u32>, 16>;
` + NOISE_WGSL + TERRAIN_FUNCS + clipSampleWGSL(filterA) + FOREST_FUNCS + TREE_CONSTS + /* wgsl */ `
fn terrainShadow(p: vec3f, L: vec3f) -> f32 {
  if (L.y > 0.8) { return 1.0; }
  let ceiling = waterLevel(p.z) + F.valley.w + F.relief.x * (1.5 - F.biome.z) + 40.0;
  var t = 30.0;
  var res = 1.0;
  for (var i = 0; i < 8; i++) {
    let q = p + L * t;
    if (q.y > ceiling) { break; }
    let texel = max(t * 0.22, 24.0);
    let lf = clamp(log2(texel / CLIP_T0), 0.0, f32(CLIP_L - 1));
    let h = clipHeight(q.xz, ceil(lf));
    let gap = q.y - h;
    res = min(res, gap / (t * 0.05) + 0.5);
    if (res <= 0.0) { return 0.0; }
    t *= 1.72;
  }
  return smoothstep(0.0, 1.0, res);
}
fn cloudShade(world: vec3f) -> f32 {
  if (F.cloudP.w <= 0.001 || F.sunDir.y <= 0.0) { return 1.0; }
  let uv = (world.xz - SH.origin) / SH.span;
  let edge = smoothstep(0.0, 0.03, uv.x) * smoothstep(0.0, 0.03, uv.y)
           * smoothstep(0.0, 0.03, 1.0 - uv.x) * smoothstep(0.0, 0.03, 1.0 - uv.y);
  let s = textureSampleLevel(cloudShadowTex, sampC, clamp(uv, vec2f(0.001), vec2f(0.999)), 0.0).r;
  return mix(1.0, s, edge);
}

@compute @workgroup_size(${TREE_GEN_GROUP}, ${TREE_GEN_GROUP}, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= G.dim || id.y >= G.dim) { return; }
  if (F.forest.x <= 0.0) { return; }
  let cell = G.cell0 + vec2i(id.xy);
  let sp = F.forest.y;
  // trunk position and the cheap range test first: most cells stop here
  let h0 = pcgHash((u32(cell.x) * 73856093u) ^ (u32(cell.y) * 19349663u + seedU() + 0x5bd1e995u));
  let pos = (vec2f(cell) + 0.5 + (vec2f(unit16(h0), unit16(h0 >> 16u)) - 0.5) * 0.8) * sp;
  let relH = pos - F.camPos.xz;
  let d2 = dot(relH, relH);
  if (d2 > TREE_RANGE * TREE_RANGE) { return; }
  let dist = sqrt(d2);
  // stand mask first, it is one texture read
  let m = treeMasks(pos);
  if (m.x < 0.02) { return; }
  let t = treeCell(cell, false, m.x, m.y);
  if (!t.ok) { return; }
  // Ground under the trunk, at the same continuous level the terrain mesh uses. A stepped
  // level made every tree grow or sink as the level boundary swept past it.
  let d3 = distance(vec3f(pos.x, F.lodP.z, pos.y), F.camPos.xyz);
  let lod = clamp(ceil(log2(max(d3, 1.0) / F.lodP.x)), 0.0, 9.0);
  let range = F.lodP.x * exp2(lod);
  let spacing = (F.lodP.x / (16.0 * F.lodP.y)) * exp2(lod);
  let morph = clamp((d3 - 0.8 * range) / (0.2 * range), 0.0, 1.0);
  let lv = clamp(max(clipFinest(pos), log2(spacing * (1.0 + morph) / CLIP_T0)), 0.0, f32(CLIP_L - 1));
  let gy = clipHeight(pos, lv);
  if (gy <= waterLevel(pos.y) + 0.2) { return; }
  // frustum, padded by the size of the crown
  let R = max(t.r * 1.3, 2.0) + t.top * 0.5;
  let rel = vec3f(pos.x - F.camPos.x, gy + t.top * 0.5 - F.camPos.y, pos.y - F.camPos.z);
  let z = dot(rel, F.camFwd.xyz);
  if (z < -R) { return; }
  if (z > R) {
    let tx = length(F.camRight.xyz);
    let ty = length(F.camUp.xyz);
    let nx = abs(dot(rel, F.camRight.xyz)) / (tx * tx * z);
    let ny = abs(dot(rel, F.camUp.xyz)) / (ty * ty * z);
    if (nx > 1.05 + R / (z * tx) || ny > 1.05 + R / (z * ty)) { return; }
  }

  // light this tree sees
  var lt = readLight();
  let ground = vec3f(pos.x, gy, pos.y);
  let sunUp = smoothstep(-0.015, 0.02, lt.L.y);
  var sunVis = 0.0;
  var cloudSh = 1.0;
  if (sunUp > 0.0) {
    cloudSh = cloudShade(ground);
    sunVis = terrainShadow(ground + vec3f(0.0, 2.0, 0.0), lt.L) * cloudSh * sunUp;
  }
  // the stand this tree belongs to, lit the way the far ground is lit
  let V = normalize(F.camPos.xyz - vec3f(pos.x, gy + t.top * 0.6, pos.y));
  let g = acos(clamp(dot(V, lt.L), -1.0, 1.0));
  let litBase = clamp(0.22 + 0.5 * lt.L.y + 0.22 * V.y, 0.0, 1.0);
  let lit = mix(litBase, 1.0, exp(-g * 3.5));
  let side = sqrt(max(1.0 - lt.L.y * lt.L.y, 0.0)) * smoothstep(-0.02, 0.1, lt.L.y);
  let crownCos = max(lt.L.y, 0.0) * 0.75 + side * mix(0.3, 0.4, m.y);
  let up = vec3f(0.0, 1.0, 0.0);
  let stand = standAlbedoAt(pos, m.y) * (lt.sun * sunVis * lit * crownCos
            + lt.moon * max(dot(up, lt.M), 0.0) + ambient(lt, up) * 0.7) * (1.0 / PI);
  let alb = treeAlbedo(t);

  var inst: TreeInst;
  inst.a = vec4f(pos.x, gy, pos.y, t.r);
  inst.b = vec4f(t.top, t.base, t.seed, select(0.0, 1.0, t.conifer));
  inst.c = vec4f(sunVis, cloudSh * sunUp, m.x, m.y);
  inst.d = vec4f(stand, 0.0);
  inst.e = vec4f(alb, 0.0);
  // One arrangement of leaf clumps for a hardwood at every distance. Close ones are
  // smooth ellipsoids; farther ones are the same clumps with eight triangles each.
  // Firs and juniper keep the near shell and the far cone.
  if (!t.conifer && F.mist.z < 0.5) {
    if (dist < TREE_CLUMP_DETAIL) {
      let c = atomicAdd(&args[9], 1u);
      if (c < TREE_DETAIL_MAX) { trees[TREE_NEAR_MAX + c] = inst; }
    } else {
      let c = atomicAdd(&args[13], 1u);
      if (c < TREE_COARSE_MAX) { trees[TREE_NEAR_MAX + TREE_DETAIL_MAX + c] = inst; }
    }
  } else if (dist < TREE_NEAR_DIST) {
    let i = atomicAdd(&args[1], 1u);
    if (i < TREE_NEAR_MAX) { trees[i] = inst; }
  } else {
    let j = atomicAdd(&args[5], 1u);
    if (j < TREE_FAR_MAX) { trees[TREE_MAX - 1u - j] = inst; }
  }
}
`;
}

// Crowns. Near trees are a twelve-sided shell: a fir with a branch notch and a ragged skirt,
// a lobed broadleaf dome on a stem, or a juniper cushion. The far list is a cone plus a stem.
// Both are shaded from the shell normal, with leaf clumps and light through the foliage.
export function treeDrawWGSL() {
  return FRAME_WGSL + ATMO_CORE_WGSL + /* wgsl */ `
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(2) var<storage, read> trees: array<TreeInst>;
@group(0) @binding(3) var apTex: texture_3d<f32>;
@group(0) @binding(4) var sampC: sampler;
` + RAIN_STRUCT + /* wgsl */ `
@group(0) @binding(5) var<uniform> Rain: RainU;
` + NOISE_WGSL + TERRAIN_FUNCS + TREE_CONSTS + /* wgsl */ `
const AP_N: f32 = ${AP_RES}.0;
const AP_MAX: f32 = ${AP_MAX_KM}.0;
override FAR: bool = false;
override LUMP: bool = false;
override COARSE: bool = false;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) rel: vec3f,
  @location(1) dv: vec2f,
  @location(2) nrm: vec3f,
  @location(3) bark: f32,
  @location(4) yFrac: f32,
  @location(5) crownU: f32,
  @location(6) @interpolate(flat) ti: u32,
  @location(7) lump: vec2f,   // x = this clump's own brightness, y = occlusion inside the clump
};

fn lumpH(seed: f32, i: u32, k: u32) -> f32 {
  return f32(pcgHash(u32(seed) * 2654435761u + i * 40503u + k * 9973u + 17u) & 0xFFFFu) * (1.0 / 65535.0);
}

// One leaf clump of a broadleaf crown: centre (xz from the trunk, y above the ground),
// half-width and half-height, a spin for its facets, and a random brightness.
struct Lump { c: vec3f, rh: f32, rv: f32, rot: f32, rnd: f32 };

// Layers: an apex clump, a ring of three near the top, a ring of five at mid height
// and three on the skirt. The skirt leaves gaps, so the stem and the sky show through.
fn lumpOf(i: u32, r: f32, base: f32, top: f32, seed: f32, squash: f32) -> Lump {
  var layer = 0u;
  var k = 0u;
  var n = 1u;
  if (i >= 1u && i <= 3u) { layer = 1u; k = i - 1u; n = 3u; }
  else if (i >= 4u && i <= 8u) { layer = 2u; k = i - 4u; n = 5u; }
  else if (i >= 9u) { layer = 3u; k = i - 9u; n = 3u; }
  let span = max(top - base, 0.5);
  var hf = 1.0;
  var ring = 0.0;
  if (layer == 1u) { hf = 0.72; ring = 0.40; }
  else if (layer == 2u) { hf = 0.47; ring = 0.68; }
  else if (layer == 3u) { hf = 0.22; ring = 0.74; }
  let j0 = lumpH(seed, i, 0u);
  let j1 = lumpH(seed, i, 1u);
  let j2 = lumpH(seed, i, 2u);
  let a = 6.2831853 * (f32(k) + 0.35 * j0) / f32(n) + f32(layer) * 1.3 + seed * 0.37;
  let rr = r * ring * (0.9 + 0.2 * j1);
  var L: Lump;
  L.rh = r * (0.41 + 0.09 * j2) * select(1.0, 1.1, layer == 0u);
  L.rv = L.rh * mix(0.86, 0.62, squash);
  var cy = base + hf * span + (j1 - 0.5) * 0.07 * span;
  if (layer == 0u) { cy = top - L.rv; }
  L.c = vec3f(cos(a) * rr, cy, sin(a) * rr);
  L.rot = 6.2831853 * j0;
  L.rnd = j2;
  return L;
}

// (radius fraction, height fraction). Height is 1 at the apex and 0 at the crown base.
// The shell stays close to a soft cone or dome. Deep notches read as low-poly spikes
// from the air; the foliage break-up is done in the shading instead.
fn crownKnot(ring: u32, conifer: bool, open: f32, trop: f32, juniper: bool) -> vec2f {
  let s = f32(ring) * (1.0 / f32(TREE_RINGS_NEAR - 1u));
  if (juniper) {
    let z = s * 2.7;
    return vec2f(sin(min(z, 3.0)) * select(1.0, 0.85, s > 0.92), pow(1.0 - s, 1.25));
  }
  if (conifer) {
    // Fuller than a straight triangle. The skirt tucks in at the bottom so the
    // trunk shows under the branches instead of the foliage planting itself in the ground.
    let cone = pow(max(s, 0.0), 0.72);
    let tuck = mix(1.0, 0.58, smoothstep(0.58, 1.0, s));
    let scallop = 0.045 * sin(s * 12.566);
    return vec2f(cone * tuck * (1.0 + scallop), 1.0 - pow(s, 0.95));
  }
  // Closed stands stay wide where neighbours meet. An open crown is a spreading
  // oak, wider than it is tall, pinching onto the stem. A sphere on a pole reads as a toy.
  let o = max(open, trop);
  let spread = sin(3.14159265 * pow(max(s, 0.0), 0.58));
  let z = 0.03 + 0.97 * s;
  var ell = sqrt(max(1.0 - pow(2.0 * z - 1.02, 2.0), 0.0));
  if (trop > 0.01) {
    let zt = pow(s, mix(1.0, 0.62, trop));
    let umb = sqrt(max(1.0 - pow(2.0 * zt - 1.12, 2.0), 0.0));
    // The floor keeps the umbrella from tapering to a cone on a thin stem.
    ell = mix(ell, max(max(umb, ell * 0.9), 0.62), trop);
  }
  let rounded = mix(ell, spread, 0.65);
  // Closed hardwoods are an ellipsoid widest in the upper half, the way a real crown holds
  // its mass. The old paraboloid was widest at the base and read as a pod.
  let rad = mix(ell, rounded, o) * smoothstep(0.0, 0.14, s);
  let h = mix(1.0 - s, 1.0 - pow(s, mix(1.65, 1.85, trop)), o);
  return vec2f(rad, h);
}

fn crownOutline(a: f32, seed: f32, ring: u32, conifer: bool, open: f32, juniper: bool) -> f32 {
  let n1 = sin(a * 2.0 + seed * 0.11 + f32(ring) * 0.4);
  let n2 = sin(a * 5.0 + seed * 0.07 + f32(ring));
  if (conifer) { return clamp(1.0 + n1 * 0.07 + n2 * 0.04, 0.88, 1.12); }
  if (juniper) { return clamp(1.0 + n1 * 0.07 + n2 * 0.04, 0.86, 1.12); }
  // A hardwood is ragged underneath where branches and clumps break the edge, and smoother
  // on top. The fine ripple sits on the lower crown only.
  let s = f32(ring) / f32(TREE_RINGS_NEAR - 1u);
  let n3 = sin(a * 9.0 + seed * 0.31 + f32(ring) * 1.7);
  let skirt = smoothstep(0.4, 1.0, s);
  return clamp(1.0 + n1 * mix(0.10, 0.16, open) + n2 * 0.05 + n3 * 0.08 * skirt, 0.72, 1.22);
}

// Perpendicular of a chord on the crown shell, in (radial, up).
fn knotDelta(p: vec2f, q: vec2f, ang: f32, crownR: f32, span: f32) -> vec3f {
  let dRad = (q.x - p.x) * crownR;
  let dY = (q.y - p.y) * span;
  return vec3f(cos(ang) * (-dY), dRad, sin(ang) * (-dY));
}

@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var o: VOut;
  o.pos = vec4f(2.0, 2.0, 2.0, 1.0);   // outside the clip volume: the triangle is discarded
  o.rel = vec3f(0.0);
  o.dv = vec2f(0.0);
  o.nrm = vec3f(0.0, 1.0, 0.0);
  o.bark = 0.0;
  o.yFrac = 0.0;
  o.crownU = 0.0;
  o.ti = 0u;
  o.lump = vec2f(0.5, 1.0);
  var ti = ii;
  if (LUMP && COARSE) {
    if (ii >= TREE_COARSE_MAX) { return o; }
    ti = TREE_NEAR_MAX + TREE_DETAIL_MAX + ii;
  } else if (LUMP) {
    if (ii >= TREE_DETAIL_MAX) { return o; }
    ti = TREE_NEAR_MAX + ii;
  } else if (FAR) {
    if (ii >= TREE_FAR_MAX) { return o; }
    ti = TREE_MAX - 1u - ii;
  } else if (ii >= TREE_NEAR_MAX) {
    return o;
  }
  let I = trees[ti];
  let top = I.b.x;
  let base = I.b.y;
  let conifer = I.b.w > 0.5;
  let spanH = max(top - base, 0.2);
  let centre = vec3f(I.a.x, I.a.y, I.a.z);
  // True crown radius at every distance. Enlarging small crowns to a minimum pixel size
  // made them shrink back to their real width as the camera approached.
  let r = I.a.w;
  let seed = I.b.z;
  let open = 1.0 - smoothstep(0.12, 0.42, I.c.z);
  let trop = F.biome.w;
  let juniper = F.mist.z > 0.5;
  let lean = vec2f(cos(seed * 0.21), sin(seed * 0.17)) * min(top * 0.05, 2.2);
  let trunkR = clamp(top * 0.016, 0.16, 0.72);
  let tri = vi / 3u;
  let k = vi % 3u;

  var world = centre;
  var dv = vec2f(0.0);
  var nrm = vec3f(0.0, 1.0, 0.0);
  var bark = 0.0;
  var yAbs = 0.0;
  var crownU = 1.0;
  var live = false;

  if (COARSE && tri < COARSE_CROWN_TRIS) {
    live = true;
    let li = tri / COARSE_LUMP_TRIS;
    let lt = tri % COARSE_LUMP_TRIS;
    let side = lt / 2u;
    let lower = (lt & 1u) == 1u;
    let lp = lumpOf(li, r, base, top, seed, max(open * 0.6, trop));
    var th = 0.0;
    var ph = 0.0;
    if (!lower) {
      if (k == 0u) { th = 0.0; }
      else { th = 1.5707963; ph = 6.2831853 * f32(side + k - 1u) / 4.0; }
    } else if (k == 0u) {
      th = 3.14159265;
    } else {
      th = 1.5707963;
      ph = 6.2831853 * f32(side + select(1u, 0u, k == 2u)) / 4.0;
    }
    ph += lp.rot;
    let u = vec3f(sin(th) * cos(ph), cos(th), sin(th) * sin(ph));
    let wob = 1.0 + 0.10 * sin(ph * 2.0 + lp.rot * 3.0);
    let radii = vec3f(lp.rh * wob, lp.rv, lp.rh * wob);
    let off = lp.c + u * radii;
    yAbs = off.y;
    let horiz = vec2f(off.x, off.z) + lean * (yAbs / max(top, 0.5));
    world = vec3f(centre.x + horiz.x, centre.y + yAbs, centre.z + horiz.y);
    dv = horiz;
    let nEll = normalize(u / radii);
    let dc = off - vec3f(0.0, base + 0.5 * spanH, 0.0);
    let dcl = length(dc);
    var nsum = nEll;
    if (dcl > 1e-3) { nsum = nEll * 0.6 + dc / dcl * 0.4; }
    if (length(nsum) < 1e-3) { nrm = nEll; } else { nrm = normalize(nsum); }
    crownU = clamp((yAbs - base) / spanH, 0.0, 1.0);
    o.lump = vec2f(lp.rnd, mix(0.8, 1.0, smoothstep(-0.7, 0.5, u.y)));
  } else if (LUMP && tri < CLUMP_CROWN_TRIS) {
    live = true;
    let li = tri / LUMP_TRIS;
    let lt = tri % LUMP_TRIS;
    // This corner as (ring 0..3, segment): a top cap, two middle bands and a bottom cap
    var jr = 0u;
    var js = 0u;
    if (lt < LUMP_SEGS) {
      if (k == 0u) { jr = 0u; js = lt; }
      else if (k == 1u) { jr = 1u; js = lt; }
      else { jr = 1u; js = lt + 1u; }
    } else if (lt < 3u * LUMP_SEGS) {
      let q = lt - LUMP_SEGS;
      let sg = q / 2u;
      if ((q & 1u) == 0u) {
        if (k == 0u) { jr = 1u; js = sg; }
        else if (k == 1u) { jr = 2u; js = sg; }
        else { jr = 2u; js = sg + 1u; }
      } else {
        if (k == 0u) { jr = 1u; js = sg; }
        else if (k == 1u) { jr = 2u; js = sg + 1u; }
        else { jr = 1u; js = sg + 1u; }
      }
    } else {
      let sg = lt - 3u * LUMP_SEGS;
      if (k == 0u) { jr = 2u; js = sg; }
      else if (k == 1u) { jr = 3u; js = sg; }
      else { jr = 2u; js = sg + 1u; }
    }
    let lp = lumpOf(li, r, base, top, seed, max(open * 0.6, trop));
    let th = 3.14159265 * f32(jr) / 3.0;
    let ph = 6.2831853 * f32(js % LUMP_SEGS) / f32(LUMP_SEGS) + lp.rot;
    let u = vec3f(sin(th) * cos(ph), cos(th), sin(th) * sin(ph));
    // the wobble depends only on the ring and the angle, so neighbouring facets still meet
    let wob = 1.0 + 0.10 * sin(ph * 2.0 + lp.rot * 3.0 + f32(jr) * 1.1);
    let radii = vec3f(lp.rh * wob, lp.rv, lp.rh * wob);
    let off = lp.c + u * radii;
    yAbs = off.y;
    let horiz = vec2f(off.x, off.z) + lean * (yAbs / max(top, 0.5));
    world = vec3f(centre.x + horiz.x, centre.y + yAbs, centre.z + horiz.y);
    dv = horiz;
    // Each clump is shaded as a rounded lump, leaning a little toward the crown as a whole
    // so the tree still reads as one lit mass.
    let nEll = normalize(u / radii);
    let dc = off - vec3f(0.0, base + 0.5 * spanH, 0.0);
    let dcl = length(dc);
    var nsum = nEll;
    if (dcl > 1e-3) { nsum = nEll * 0.6 + dc / dcl * 0.4; }
    if (length(nsum) < 1e-3) { nrm = nEll; } else { nrm = normalize(nsum); }
    crownU = clamp((yAbs - base) / spanH, 0.0, 1.0);
    o.lump = vec2f(lp.rnd, mix(0.8, 1.0, smoothstep(-0.7, 0.5, u.y)));
  } else if (FAR) {
    if (tri < TREE_SEGS_FAR) {
      live = true;
      var si = tri;
      if (k > 0u) { si = tri + k - 1u; }
      let s = select(0.0, 1.0, k > 0u);
      let a = f32(si % TREE_SEGS_FAR) * (6.2831853 / f32(TREE_SEGS_FAR)) + seed * 0.37;
      let jr = 1.0 + 0.10 * sin(a * 3.0 + seed * 0.13);
      let rad = s * r * 1.0996 * jr;
      yAbs = mix(top, base, s);
      let horiz = vec2f(cos(a), sin(a)) * rad + lean * (yAbs / max(top, 0.5));
      world = vec3f(centre.x + horiz.x, centre.y + yAbs, centre.z + horiz.y);
      dv = horiz;
      let Rbase = r * 1.0996 * jr;
      let slope = vec3f(cos(a) * (top - base), Rbase, sin(a) * (top - base));
      let slen = length(slope);
      if (slen < 1e-4) { nrm = vec3f(0.0, 1.0, 0.0); } else { nrm = slope / slen; }
      crownU = 1.0 - s;
    } else if (base > 1.15 && !juniper) {
      live = true;
      let ft = tri - TREE_SEGS_FAR;
      let second = (ft & 1u) == 1u;
      var corner = k;
      if (second) { corner = select(0u, k + 1u, k > 0u); }
      let topC = corner >= 2u;
      yAbs = select(0.0, min(base + spanH * 0.05, top * 0.96), topC);
      let midY = base * 0.45;
      let mid = vec3f(centre.x, centre.y, centre.z) + vec3f(lean.x, 0.0, lean.y) * (midY / max(top, 0.5)) + vec3f(0.0, midY, 0.0);
      let eye = F.camPos.xyz - mid;
      let axis = vec3f(lean.x, max(top, 1.0), lean.y);
      var side = cross(axis, eye);
      if (length(side) < 1e-2) { side = vec3f(1.0, 0.0, 0.0); } else { side = normalize(side); }
      let positive = corner == 1u || corner == 2u;
      let sgn = select(-1.0, 1.0, positive);
      let hw = trunkR * select(1.35, 0.7, topC);
      let p = vec3f(centre.x, centre.y + yAbs, centre.z) + vec3f(lean.x, 0.0, lean.y) * (yAbs / max(top, 0.5));
      world = p + side * hw * sgn;
      dv = vec2f(sgn * hw, yAbs);
      let eh = vec3f(eye.x, 0.22 * length(eye.xz), eye.z);
      if (length(eh) < 1e-3) { nrm = vec3f(0.0, 1.0, 0.0); } else { nrm = normalize(eh); }
      bark = 1.0;
      crownU = 0.0;
    }
  } else if (!LUMP && tri < NEAR_CROWN_TRIS) {
    live = true;
    let bandTris = TREE_SEGS_NEAR * 2u;
    let band = tri / bandTris;
    let bt = tri % bandTris;
    let seg = bt / 2u;
    let second = (bt & 1u) == 1u;
    var ring = band;
    var si = seg;
    if (!second) {
      if (k == 0u) { ring = band; si = seg; }
      else if (k == 1u) { ring = band + 1u; si = seg; }
      else { ring = band + 1u; si = seg + 1u; }
    } else {
      if (k == 0u) { ring = band; si = seg; }
      else if (k == 1u) { ring = band + 1u; si = seg + 1u; }
      else { ring = band; si = seg + 1u; }
    }
    let a = f32(si % TREE_SEGS_NEAR) * (6.2831853 / f32(TREE_SEGS_NEAR)) + seed * 0.37 + f32(ring) * 0.47;
    let shape = crownKnot(ring, conifer, open, trop, juniper);
    let jr = crownOutline(a, seed, ring, conifer, open, juniper);
    let rad = shape.x * r * jr * 1.023;
    yAbs = base + shape.y * (top - base);
    var horiz = vec2f(cos(a), sin(a)) * rad + lean * (yAbs / max(top, 0.5));
    if (conifer && ring <= 1u) {
      horiz += vec2f(cos(seed * 0.9), sin(seed * 0.9)) * r * 0.04 * select(1.0, 0.45, ring == 1u);
    }
    world = vec3f(centre.x + horiz.x, centre.y + yAbs, centre.z + horiz.y);
    dv = horiz;
    var acc = vec3f(0.0);
    let shellR = r * jr * 1.023;
    if (ring > 0u) { acc += knotDelta(crownKnot(ring - 1u, conifer, open, trop, juniper), shape, a, shellR, top - base); }
    if (ring + 1u < TREE_RINGS_NEAR) { acc += knotDelta(shape, crownKnot(ring + 1u, conifer, open, trop, juniper), a, shellR, top - base); }
    if (length(acc) < 1e-3) { nrm = vec3f(0.0, 1.0, 0.0); } else { nrm = normalize(acc); }
    crownU = shape.y;
  } else if (base > 1.15 && !juniper) {
    live = true;
    let ttri = tri - select(select(NEAR_CROWN_TRIS, CLUMP_CROWN_TRIS, LUMP), COARSE_CROWN_TRIS, COARSE);
    let side = ttri / 2u;
    let second = (ttri & 1u) == 1u;
    var corner = k;
    if (second) { corner = select(0u, k + 1u, k > 0u); }
    let topC = corner >= 2u;
    let useA1 = corner == 1u || corner == 2u;
    let stepA = 6.2831853 / f32(TREE_TRUNK_SIDES);
    let a0 = (f32(side) + seed * 0.013) * stepA;
    let ang = select(a0, a0 + stepA, useA1);
    let am = a0 + stepA * 0.5;
    // under leaf clumps the stem runs up into the crown, so it shows through the skirt gaps
    yAbs = select(0.0, min(base + spanH * select(0.04, 0.5, LUMP), top * 0.97), topC);
    let radT = trunkR * select(1.32, 0.6, topC);
    let horiz = vec2f(cos(ang), sin(ang)) * radT + lean * (yAbs / max(top, 0.5));
    world = vec3f(centre.x + horiz.x, centre.y + yAbs, centre.z + horiz.y);
    dv = horiz;
    nrm = normalize(vec3f(cos(am), 0.14, sin(am)));
    bark = 1.0;
    crownU = 0.0;
  }

  if (!live) { return o; }
  var rel = world - F.camPos.xyz;
  rel.y -= dot(rel.xz, rel.xz) / (2.0 * EARTH_R);
  o.rel = rel;
  o.pos = F.viewProj * vec4f(rel, 1.0);
  o.dv = dv;
  o.nrm = nrm;
  o.bark = bark;
  o.yFrac = yAbs / max(top, 0.5);
  o.crownU = crownU;
  o.ti = ti;
  return o;
}

fn shadeSurface(alb: vec3f, nIn: vec3f, V: vec3f, lt: Light, shade: f32, ao: f32, transK: f32, specK: f32, rough: f32) -> vec3f {
  var n = nIn;
  let nlen = length(n);
  // A NaN length fails the comparison, so the test is inverted.
  if (!(nlen > 1e-4)) { n = vec3f(0.0, 1.0, 0.0); } else { n = n / nlen; }
  // A normal that faces the camera by a hair more than 1 makes pow() return NaN.
  // Those pixels survive the bloom chain and show up as white spots in the sky.
  let ndv = clamp(dot(n, V), -1.0, 1.0);
  if (ndv < 0.0) { n = -n; }
  let facing = abs(ndv);
  let back = ndv < 0.0;
  let sunHit = max(dot(n, lt.L), 0.0) * select(1.0, 0.22, back);
  let moonHit = max(dot(n, lt.M), 0.0);
  let rim = pow(clamp(1.0 - facing, 0.0, 1.0), 1.8);
  let trans = transK * max(-dot(n, lt.L), 0.0) * (0.28 + 0.72 * rim);
  let Hl = V + lt.L;
  let Hlen = length(Hl);
  var spec = 0.0;
  if (Hlen > 1e-4 && specK > 0.0) {
    let nh = clamp(dot(n, Hl / Hlen), 0.0, 1.0);
    spec = specK * ggx(nh, max(rough, 0.08)) * sunHit / max(facing, 0.2);
  }
  let amb = ambient(lt, n) * ao * select(1.0, 0.5, back);
  return (alb * (lt.sun * shade * (sunHit + trans) + lt.moon * moonHit + amb)
       + lt.sun * shade * spec * PI) * (1.0 / PI);
}

@fragment fn fs(in: VOut) -> @location(0) vec4f {
  let I = trees[in.ti];
  let dist = max(length(in.rel), 0.05);
  let V = -in.rel / dist;
  let conifer = I.b.w > 0.5;
  let top = I.b.x;
  let r = I.a.w;
  let seed = I.b.z;
  let sunVis = I.c.x;
  let cloudSu = I.c.y;
  let dens = I.c.z;
  var lt = readLight();

  // Below a pixel the crown is the stand colour the ground already carries.
  let mpp = max(length(dpdx(in.rel)), length(dpdy(in.rel)));
  let kS = smoothstep(0.55 * r, 2.3 * r, mpp);
  var col = I.d.xyz;
  if (kS < 0.999) {
    let yAbs = in.yFrac * top;
    let sunk = smoothstep(-0.02, 0.35, lt.L.y);
    let shDepth = smoothstep(max(F.forest.z, 1.0) * 0.95, max(F.forest.z, 1.0) * 0.18, yAbs) * smoothstep(0.05, 0.5, dens);
    let sh = select(1.0, clamp(1.0 - shDepth * mix(0.82, 0.36, sunk), 0.0, 1.0), dens > 0.004);
    let kd = 1.0 - smoothstep(0.05 * r, 0.42 * r, mpp);
    var n = in.nrm;
    if (length(n) < 1e-4) { n = vec3f(0.0, 1.0, 0.0); } else { n = normalize(n); }
    var own = vec3f(0.0);
    if (in.bark > 0.5) {
      let ang = atan2(in.dv.y, in.dv.x);
      let g = snoise(vec2f(ang * 2.2 + seed * 0.01, yAbs * 1.6));
      let ridge = snoise(vec2f(ang * 8.0, yAbs * 3.2 + seed));
      var barkC = select(vec3f(0.072, 0.054, 0.038), vec3f(0.064, 0.038, 0.024), conifer);
      barkC *= 0.78 + 0.30 * (0.5 + 0.5 * g);
      barkC *= mix(0.58, 1.0, smoothstep(-0.3, 0.6, ridge));
      if (F.biome.w > 0.5 || F.forest.z > 36.0) {
        let moss = smoothstep(0.28, 0.02, in.yFrac) * (0.35 + 0.65 * max(g, 0.0));
        barkC = mix(barkC, vec3f(0.016, 0.034, 0.015), moss * 0.8);
      }
      let aoB = mix(0.48, 1.0, smoothstep(0.0, 0.45, in.yFrac));
      own = shadeSurface(barkC, n, V, lt, sh * sunVis * 0.85, aoB, 0.0, 0.004, 0.72);
    } else {
      // Broadleaf foliage is sampled by position, not by angle around the trunk: each leaf
      // clump is its own lump, and an angle has a seam and collapses at the trunk axis.
      let wy = in.yFrac * top;
      let wp = vec2f(in.dv.x + wy * 0.4, in.dv.y - wy * 0.35) * (2.4 / max(r, 1.0));
      if (kd > 0.03) {
        let ang = atan2(in.dv.y, in.dv.x);
        let qc = vec2f(ang * 1.1 + length(in.dv) * 0.08, in.crownU * 2.6);
        let q = select(wp, qc, conifer);
        let q2 = q + vec2f(seed * 0.017, seed * 0.008);
        let b1 = gnoised(q2);
        var side = vec3f(-in.dv.y, 0.0, in.dv.x);
        if (length(side) < 1e-3) { side = vec3f(1.0, 0.0, 0.0); } else { side = normalize(side); }
        let amp = select(0.38, 0.5, conifer) * kd;
        let bumped = n + side * b1.y * amp + vec3f(0.0, 1.0, 0.0) * b1.x * amp * 0.25;
        let blen = length(bumped);
        if (!(blen > 1e-4)) { n = vec3f(0.0, 1.0, 0.0); } else { n = bumped / blen; }
      }
      var alb = I.e.xyz;
      let x = clamp(length(in.dv) / max(r, 0.3), 0.0, 1.35);
      if (kd > 0.02) {
        let ang = atan2(in.dv.y, in.dv.x);
        let clA = snoise(vec2f(ang * 0.55, in.crownU * 1.6) + vec2f(seed * 0.02, 1.7));
        let clB = snoise(vec2f(ang * 1.4 + x, in.crownU * 2.6) + vec2f(4.0, seed * 0.01));
        let clC = snoise(wp * 0.5 + vec2f(seed * 0.02, 1.7));
        let clD = snoise(wp * 1.3 + vec2f(4.0, seed * 0.01));
        let cl = select(clC, clA, conifer);
        let cl2 = select(clD, clB, conifer);
        alb *= 1.0 + 0.2 * cl + 0.1 * cl2;
      }
      // each leaf clump has its own brightness (0.5 off the clump mesh leaves it unchanged)
      alb *= 0.92 + 0.16 * in.lump.x;
      // A little darker toward the stem. Kept mild: a strong dark base reads as a hole
      // in the canopy and breaks a closed stand into separate caps.
      alb *= mix(0.9, 1.04, smoothstep(0.0, 0.75, in.crownU));
      if (conifer) {
        alb *= mix(0.9, 1.05, smoothstep(0.05, 0.85, x));
      } else {
        alb *= mix(0.92, 1.03, smoothstep(0.0, 0.7, x));
      }
      let ao = (0.5 + 0.5 * in.crownU) * mix(0.78, 1.05, smoothstep(-0.2, 0.65, n.y)) * in.lump.y;
      let transK = select(0.26, 0.48, conifer);
      let specK = select(0.015, 0.005, conifer);
      own = shadeSurface(alb, n, V, lt, sh * sunVis, clamp(ao, 0.35, 1.15), transK, specK, select(0.46, 0.6, conifer));
    }
    col = mix(own, col, kS);
  }
  col *= mix(1.0, 0.72, Rain.amt.x);

  // valley mist, precipitation and aerial perspective, as for the ground
  let xz = vec2f(I.a.x, I.a.z) + in.dv;
  let wl = waterLevel(xz.y);
  if (F.mist.x > 0.0) {
    let topM = wl + F.mist.y;
    let py = F.camPos.y + in.rel.y;
    let below = max(topM - py, 0.0);
    if (below > 0.0) {
      let pn = fbm2(xz * (1.0 / 520.0) + vec2f(F.camPos.w * 0.004, 1.0), 2) * 0.5 + 0.5;
      let thick = F.mist.x * 0.0022 * smoothstep(0.5, 0.85, pn) * smoothstep(0.0, F.mist.y * 0.6, below);
      let camIn = max(topM - F.camPos.y, 0.0);
      let path = min((below - camIn) / max(V.y, 0.04), dist);
      let Tm = exp(-thick * max(path, 0.0));
      let cosT = dot(-V, lt.L);
      let ph = mix(phaseHG(cosT, 0.6), phaseHG(cosT, -0.1), 0.5) * 4.0 * PI;
      let mistCol = lt.sun * cloudSu * ph * 0.16 + (lt.zen * 0.6 + lt.hor * 0.4) * 0.9 + lt.moon * 0.3;
      col = col * Tm + mistCol * (1.0 - Tm);
    }
  }
  if (Rain.wind.w > 0.004) {
    let snow = Rain.amt.y;
    let thick = Rain.wind.w * mix(0.00042, 0.00018, snow);
    let Tm = exp(-thick * dist);
    let fogCol = mix(vec3f(0.58, 0.63, 0.7), vec3f(0.86, 0.89, 0.93), snow);
    var litFog = (lt.zen * 0.62 + lt.hor * 0.38) * mix(0.5, 0.85, snow);
    litFog += vec3f(0.62, 0.72, 0.95) * Rain.amt.z * 0.9;
    col = col * Tm + fogCol * litFog * (1.0 - Tm);
    if (Rain.bolt.w > 0.5) { col += vec3f(0.42, 0.52, 0.78) * Rain.amt.z * 0.6; }
  }
  let uv = in.pos.xy * F.screen.zw;
  let zc = sqrt(dist * 0.001 / AP_MAX) * AP_N - 1.0;
  let w = clamp(zc + 1.0, 0.0, 1.0);
  let ap = textureSampleLevel(apTex, sampC, vec3f(uv, (max(zc, 0.0) + 0.5) / AP_N), 0.0);
  let T = mix(1.0, ap.a, w);
  col = col * T + ap.rgb * F.sunDir.w * w;
  // max() with a finite value replaces NaN, so a bad leaf never blooms into a sky spot.
  col = min(max(col, vec3f(0.0)), vec3f(64.0));
  return vec4f(col, 1.0);
}
`;
}
