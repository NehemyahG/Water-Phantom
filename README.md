# Synthetic proton water phantom: DICOM CT + RTSTRUCT + OpenTPS

**Research/testing only.** This creates **DICOM CT images and an RTSTRUCT**, with explicit patient-space coordinates and a programmable target. OpenTPS can calculate a static proton dose using (A) an illustrative plan it constructs, or (B) a previously exported proton PBS RTPLAN. The package **does not yet simulate a physical ridge filter or FLASH dose rate**.

## Default geometry

| Parameter | Default | Change with |
|---|---|---|
| Phantom transverse cross-section | 10 × 10 cm | `--cross-section-mm 100 100` |
| Water depth | 20 cm along patient **y** | `--depth-mm 200 --depth-axis y` |
| Water dimensions (LPS x,y,z) | (100, 200, 100) mm | `--water-mm X Y Z` |
| Surrounding air border | 50 mm on every side | `--air-margin-mm 50` |
| CT FOV (LPS x,y,z) | (200, 300, 200) mm | `--fov-mm X Y Z` |
| CT voxel spacing (x,y,z) | (2, 2, 2) mm | `--spacing-mm DX DY DZ` |
| CT slices | 100 axial slices (150 rows × 100 columns) | determined by size and spacing |
| Water / air | 0 / −1000 HU | `--water-hu` and `--air-hu` |
| External ROI | exactly the water cuboid | automatic RTSTRUCT |
| Target ROI | 60 × 40 × 60 mm centered at isocenter | `--target-mm X Y Z`, `--target-offset-mm X Y Z` |
| Isocenter LPS | (0, 0, 0) mm | `--isocenter-mm X Y Z` or `--rtplan` |

Why default depth axis **y**? This gives 20 cm along the approximately anterior-to-posterior direction for a demonstration **gantry 0°**/couch 0° beam. **Verify beam direction in your treatment-planning software**. DICOM axis *y* means patient posterior positive, but IEC gantry/couch geometry must be verified per beam model and plan. A beam arriving from another direction may traverse only 10 cm. If you want the conventional “10×10 cm axial cross-section and 20 cm superior-inferior length,” use `--depth-axis z`, **and separately configure a matching beam direction**.

The full physical voxel array includes air around the water so that the water surface and interface are observable. The planned target is **not** a material insert: its voxels remain water, and the contour represents a proposed dose-scoring/optimization volume.

## Installation

Use ordinary Python for phantom generation and analysis:

```bash
python -m pip install -r requirements.txt
```

For tests, also install `pytest`.

For dose calculation, use a **separate compatible OpenTPS environment** with MCsquare installed/configured and an appropriate scanner conversion table and Beam Data Library (BDL). Current OpenTPS releases may require Python 3.12, rather than the Python version used for DICOM-only generation. See [OpenTPS installation](https://gitlab.com/openmcsquare/opentps) and the [OpenTPS static-dose example](https://opentps.github.io/examples/auto_examples/DoseComputation/run_SimpleDoseCalculation.html).

## 1. Generate the CT and DICOM RTSTRUCT

```bash
python generate_water_phantom.py --out phantom_10x10x20cm
python validate_water_phantom.py --ct-dir phantom_10x10x20cm
```

Files in the output folder:

- `CT_0001.dcm` … `CT_0100.dcm`: axial HFS DICOM CT series.
- `RS_WATER_PHANTOM.dcm`: DICOM RTSTRUCT defining **External** (water volume) and **Target** (PTV-class demonstration target).
- `phantom_geometry.json`: dimensions, reference UIDs, exact voxel ranges, isocenter and ROI bounds.

The structure set references the **CT StudyInstanceUID, SeriesInstanceUID, FrameOfReferenceUID and individual CT SOPInstanceUIDs**. Each contour is a closed four-corner polygon placed on a CT slice plane, and each contour references that slice.

**Examples of changing the geometry**

```bash
# 10 x 10 cm transverse, 30 cm of water depth, y-directed:
python generate_water_phantom.py --out phantom_deep --depth-mm 300

# 10 x 10 cm cross-section perpendicular to z, 20 cm long in z:
python generate_water_phantom.py --out phantom_depth_z --depth-axis z

# Smaller voxels for improved surface-dose / interface resolution:
python generate_water_phantom.py --out phantom_1mm --spacing-mm 1 1 1

# Customized distal test target, y=20..60 mm -> nominal 120..160 mm water depth
# for a beam traveling in +y with water centered on isocenter:
python generate_water_phantom.py --out phantom_distal \
  --target-mm 60 40 60 --target-offset-mm 0 40 0
```

The voxel grid must represent every requested length exactly. For a centered ROI, the FOV minus ROI voxel counts must have even parity; offsets must be integer multiples of voxel spacing. The program rejects inconsistent geometry instead of silently shifting structures.

## 2. Verify HU calibration and density/material mapping

**0 HU is a synthetic label**, not an assurance that MCsquare interprets that voxel as physical liquid water. MCsquare converts CT HU to density and a discrete material using scanner calibration files. The separate material assignment is important: density 1 g/cm³ alone does not establish proton stopping power or nuclear scattering.

In the local scanner calibration directory, the key files are `HU_Density_Conversion.txt` and `HU_Material_Conversion.txt`. The audit prints water/air density and material IDs, and can optionally check the RSP if OpenTPS exposes `convertHU2RSP`:

```bash
python check_calibration.py --ct-dir phantom_10x10x20cm \
  --scanner-folder /path/to/MCsquare/Scanners/YOUR_SCANNER
```

If you know the water and air material IDs from the corresponding **MCsquare `Materials/list.dat`**, provide them for stricter ID checks:

```bash
python check_calibration.py --ct-dir phantom_10x10x20cm \
  --scanner-folder /path/to/MCsquare/Scanners/YOUR_SCANNER \
  --water-material-id 17 --air-material-id 3
```

**The IDs 17/3 above are illustrative placeholders, NOT recommended values.** Replace with the IDs from your installed MCsquare material library. The audit writes `calibration_audit.json` with the conversion-table SHA256 hashes and results. You may adjust the simulated HU with `--water-hu VALUE`, but doing so should be based on an established calibration/material assignment rather than arbitrary trial and error.

The External ROI includes DICOM annotations `REL_MASS_DENSITY=1.0` and `REL_STOP_RATIO=1.0`. **They are annotations only:** do not assume RayStation or MCsquare automatically applies them as overrides. Verify assignment explicitly in **each TPS**; RayStation's commissioned HU-to-density/SPR curve is separate from MCsquare's.

## 3A. Create and calculate a demonstration plan **inside OpenTPS**

In your OpenTPS environment:

```bash
python simulate_open_tps.py --ct-dir phantom_10x10x20cm \
  --out results_demo --nb-primaries 1000000
```

The script imports the generated CT, verifies the LPS origin/spacing/frame of reference, builds a **conventional, multi-energy proton PBS demonstration plan** against the same target volume described in the RTSTRUCT, and calculates a static dose using MCsquare. It saves `dose_result.npz` (dose, origin, spacing, CT HU) and `dose_run.json`.

You can override machine model paths and demonstration geometry:

```bash
python simulate_open_tps.py --ct-dir phantom_10x10x20cm \
  --scanner-folder /path/to/ScannerFolder \
  --bdl-file /path/to/beam_model.txt \
  --gantry-angle 0 --couch-angle 0 \
  --spot-spacing-mm 5 --layer-spacing-mm 5 --target-margin-mm 3
```

This is an illustrative conventional **multi-energy** plan, **not** a single-energy ridge-filter FLASH plan. Do not use the generated demo as a deliverable treatment plan.

## 3B. Create a plan in RayStation, then import it into OpenTPS

1. Import **both the CT slices and `RS_WATER_PHANTOM.dcm`** into a **research/test patient database** in RayStation. Inspect axial/coronal/sagittal planes, External/Target contours, patient position, image calibration, CT-to-density conversion and isocenter.
2. Create a **new proton plan on this phantom in RayStation**, with a machine/beam model appropriate for the research study. Export its DICOM RTPLAN.
3. Create a newly aligned synthetic CT+RTSTRUCT by using the RTPLAN **isocenter** and **FrameOfReferenceUID** (on a **new** output folder):

   ```bash
   python generate_water_phantom.py --rtplan RP_Research.dcm --out phantom_matched_to_RP
   python validate_water_phantom.py --ct-dir phantom_matched_to_RP --rtplan RP_Research.dcm
   ```

4. Compute on that aligned phantom in OpenTPS:

   ```bash
   python simulate_open_tps.py --ct-dir phantom_matched_to_RP \
     --rtplan RP_Research.dcm --out results_imported
   ```

**Important caveat about step 3:** matching patient ID, study UID, frame UID and isocenter is **necessary but not sufficient** for complete DICOM plan-transfer compatibility. An exported RTPLAN commonly references its **original RTSTRUCT SOPInstanceUID**. The newly generated RTSTRUCT has a **different UID**; this script **does not rewire or modify the exported RTPLAN**. If RayStation must consume the plan alongside the new structure set, recreate/reassociate the plan using approved research workflows. For the cleanest RayStation workflow, **plan directly on the generated CT/RTSTRUCT first** rather than backfilling images against an existing plan. Also validate all beam geometry, plan accessories, machine models, range and density assignments.

If your existing research plan already uses a different coordinate system or orientation (not HFS), the program stops rather than silently misregistering geometry. Plans with multiple unique isocenters are similarly rejected.

## 4. Extract candidate depth-dose profiles for future ridge-filter comparisons

Once a dose calculation completes:

```bash
python analyze_depth_dose.py --ct-dir phantom_10x10x20cm \
  --dose-npz results_demo/dose_result.npz \
  --axis y --entry negative --radius-mm 5 \
  --plateau-mm 80 120
```

This produces `central_axis_depth_dose.csv`: water depth in mm, mean static dose in a small transverse circle, and relative dose normalized to the profile peak. With `--plateau-mm`, it reports a candidate plateau peak-to-valley difference divided by its mean. **Only use this physical-depth convention when the beam is known to travel along the chosen axis**; it is not meaningful for arbitrary oblique gantry/couch orientations. Dose units/normalization should be confirmed in the OpenTPS output and beam model.

## Why these files are only the **first** step toward FLASH ridge filters

A ridge filter converts a **single incident energy** into a distributed spectrum via different **water-equivalent thicknesses (WET)**. It can create an SOBP without slow energy-layer switching. However, generating the **water phantom** does **not** add the upstream filter's geometry, material, physical energy loss, scattering, transmission, or time structure to the dose engine. A default multi-energy PBS plan that gives a flat SOBP **does not demonstrate** a FLASH-capable ridge-filter design.

For the next stage, you will need at least:

- **Beam source:** nominal incident energy (e.g., 150–230 MeV), energy spread, spot/field size, angular divergence, current/charge and delivery-time structure.
- **Beamline geometry:** source → ridge filter → air gap → water surface; any range shifters, scatterers, apertures and snout components.
- **Ridge filter:** material/density/relative stopping power, pitch, height profile, lateral dimensions and manufacturable minimum feature thickness.
- **Scoring:** pristine Bragg curve, SOBP width/flatness, distal falloff, lateral uniformity, transmission losses, entrance dose, uncertainties, and **dose rate as a function of time and position**.
- **Validation:** a commissioned beam model, independent Monte Carlo (e.g., TOPAS/Geant4 for full beamline), and measurement-based benchmarking before use.

In particular, at **230 MeV** the proton water range is approximately **33 cm**, so a **20-cm water depth alone cannot capture the pristine distal Bragg peak** without substantial upstream range degradation or lower energy. Test a **40-cm water depth** for 230-MeV pristine-beam studies:

```bash
python generate_water_phantom.py --out phantom_40cm --depth-mm 400
```

This distinction is essential for a FLASH project. This package enables **consistent CT/RTSTRUCT coordinates and a baseline static-dose workflow**, not the physical ridge-filter dose-rate simulation itself.

## DICOM geometry sanity check

- Patient coordinates are **LPS** in mm: **+x left, +y posterior, +z head**.
- `ImageOrientationPatient = [1,0,0,0,1,0]`: **increasing DICOM column index moves +x** and **increasing row index moves +y**; the pixel array is indexed `[row=y, column=x]`.
- `PixelSpacing = [dy, dx]`, not `[dx, dy]`.
- The first pixel center in the first CT slice is `ImagePositionPatient = (-99, -149, -99) mm` for the default phantom/isocenter. The water cuboid occupies **x = −50…50 mm**, **y = −100…100 mm**, **z = −50…50 mm** in voxel-edge boundaries.
- `IsocenterPosition` is an **RTPLAN** property. The CT's geometric registration is encoded by pixel position/orientation and `FrameOfReferenceUID`; merely copying a numeric isocenter does not certify compatibility.

## Tests and known limitations

```bash
python -m pip install pytest
python -m pytest -q tests
```

The local unit tests include calibration-table checking, depth-dose CSV, CT/RTSTRUCT roundtrip/UID validation and matching a synthetic ion RTPLAN. Run them in a Python environment with pydicom available. **External vendor import and MCsquare dose engine execution have not been run in this delivery environment.** Validate them locally before interpreting dosimetric results.

References: [DICOM RTSTRUCT specification](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.8.5.html), [MCsquare CT calibration](https://www.openmcsquare.org/documentation_CT_calibration.html), [OpenTPS examples](https://opentps.github.io/examples/auto_examples/DoseComputation/run_SimpleDoseCalculation.html), [proton water ranges from NIST PSTAR](https://www.hep.ucl.ac.uk/pbt/wiki/Proton_ranges), [proton FLASH ridge-filter beamline modeling](https://pmc.ncbi.nlm.nih.gov/articles/PMC13641327/).
