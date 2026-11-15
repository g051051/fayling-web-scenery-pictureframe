# Moon terrain

Under Settings → Scenery, set **Scenes** to **One region**.
Then select **Moon · Copernicus** in **Region**.
Alternatively, open `scenery.html?scene=moon` or `http://localhost:8081/?scene=moon`.
The tour includes the Moon after the nine Earth regions.

The camera follows seven waypoints from the southern rim, across the floor, past the central peaks, and over the northern rim.
Smooth curves connect the waypoints without sharp heading changes.
Target clearance decreases from 5.5 km on approach to 2.4 km over the floor, then increases for the exit.
The camera pitch changes gradually along the route. The Flight altitude and Camera tilt controls still apply.
The terrain probes control camera clearance with the same elevation samples that supply the mesh.
Lunar probes extend about 13 km ahead along the curved route to anticipate rising terrain.
At the route endpoint, the scene fades into the next tour scene or restarts in single-scene mode.
At Speed 1, a tour completes the route in approximately the selected scene duration, plus fades.
Single-scene mode cruises at 180 m/s and completes the full route before restarting.
The Speed control changes these travel times. Free flight suspends the route.
Autopilot rejoins at the current north–south position and eases lateral offsets back toward the route.
Free flight stays inside the measured crop.

## Elevation source

The elevation source is [SLDEM2015 from NASA/GSFC and the MIT LOLA archive](https://pgda.gsfc.nasa.gov/products/54).
This product combines LRO laser altimetry with SELENE/Kaguya stereo terrain measurements.
The source [PDS label](https://imbrium.mit.edu/DATA/SLDEM2015/TILES/FLOAT_IMG/SLDEM2015_256_0N_60N_240_360_FLOAT.LBL) defines the units, byte order, dimensions, and reference radius.

| Property | Value |
|---|---|
| Source product | `SLDEM2015_256_0N_60N_240_360_FLOAT` |
| Source grid | 256 samples per degree, about 118 m north–south |
| Crop bounds | 5–15° N, 335–345° E |
| Bundled grid | 1280 × 1280 samples, about 237 m north–south |
| Horizontal extent | Approximately 299 × 303 km |
| Height reference | Metres above a 1,737,400 m sphere |
| Encoding | Gzip-compressed, little-endian signed 16-bit integer metres |
| Processing | Average each 2 × 2 source block, then round to the nearest metre |
| Orientation | Rows run north to south. Columns run west to east. |

Citation: Barker, M. K. et al. (2016), “A new lunar digital elevation model from the Lunar Orbiter Laser Altimeter and SELENE Terrain Camera.”
*Icarus*, 273, 346–355. [DOI: 10.1016/j.icarus.2015.07.039](https://doi.org/10.1016/j.icarus.2015.07.039).

The generated `js/moon-data.js` records source URLs, bounds, elevation limits, and SHA-256 checksums.
The application decompresses this embedded asset locally.
Both HTML bundles contain the asset and require no terrain requests at runtime.

## Visual limits

The mesh uses a local equirectangular approximation centred at 10° N, 340° E.
The renderer applies lunar curvature to the camera-relative mesh.
This view is a regional flyover, not a global globe or navigation instrument.
Terrain outside the crop is not visible. Free flight near the crop boundary can expose its edge.

Elevations have no vertical exaggeration or added procedural craters.
The renderer reconstructs heights with a cubic B-spline across each 4 × 4 sample neighbourhood.
This smooths slopes between source cells and slightly softens peaks without exceeding the local elevation range.
Surface normals and camera clearance use the same reconstructed height field.
Shadow sampling blends adjacent terrain detail levels to reduce abrupt transitions.
The grey regolith colour and fine colour variations are artistic, not measured surface imagery.
The light model includes a small constant bounce-light term for visibility in shadows.
The daylight sky is black, without clouds, atmospheric scattering, stars, or a second Moon.
The Moon uses a fixed low Sun. Earth weather and astronomy do not drive this scene.
Rain, snow, vegetation, water, and the Earth weather widget are absent.
The panel hides Earth-only environment controls during the lunar scene.
The clock still shows the viewer's local time.

## Rebuild the asset

Run the preparation script from the project directory:

```powershell
python tools/prepare-moon.py
npm run build
```

The script uses the Python standard library and downloads about 315 MB through HTTP byte ranges.
It checks range responses and rejects invalid elevation values.
The cache lives under `tools/.moon-cache/`, which Git ignores.
The generated JavaScript asset is about 3 MB.

## Validation

Run the Windows integration check:

```powershell
node tools/check-scenes.mjs
```

The check starts its own local server and headless Chrome.
It checks elevation decoding, geographic bounds, shader errors, offline bundles, and a transition from the Moon to Earth.
It saves images under `shots/` for visual inspection.
Set `CHROME_PATH` if Chrome is not installed at its default Windows location.

## Other bodies

Scene metadata carries the body radius, elevation limits, and crop extent.
The height bake and camera probes share one DEM sampling function.
Additional bodies need their own asset and an explicit source selection in that sampling path.

| Body | Public source | Additional work |
|---|---|---|
| Mars | [USGS MOLA/HRSC 200 m DEM](https://astrogeology.usgs.gov/search/map/mars_mgs_mola_mex_hrsc_blended_dem_global_200m) | Dusty atmosphere, surface colour, and body-specific lighting |
| Mercury | [USGS MESSENGER 665 m DEM](https://astrogeology.usgs.gov/search/map/mercury_messenger_global_dem_665m) | New terrain crop and solar illumination scale |
| Venus | [USGS Magellan 4.6 km topography](https://astrogeology.usgs.gov/search/map/venus_magellan_global_topography_4641m) | Dense atmosphere and a higher-altitude route for the coarse elevation grid |

These bodies are not selectable scenes in this version.
