"""Compute research-only *static* proton dose on a DICOM water phantom.

Workflow A: OpenTPS plans a conventional illustrative multi-energy PBS field
    python simulate_open_tps.py --ct-dir water_phantom_CT --out demo_dose

Workflow B: OpenTPS calculates an imported RayStation PBS proton RTPLAN
    python simulate_open_tps.py --ct-dir matched_CT --rtplan RP_example.dcm

No ridge filter, upstream beamline, physical dose-rate or FLASH modeling is
provided by this script. Beam model and scanner must be commissioned for the
intended research geometry and independently validated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pydicom


def _target_mask(manifest: dict, ct) -> np.ndarray:
    target = next(r for r in manifest['roi_specs'] if r['name'] == 'Target')
    mask = np.zeros(ct.imageArray.shape, dtype=bool)
    xs, ys, zs = [slice(int(lo), int(hi))
                  for lo, hi in target['voxel_slices_xyz']]
    mask[xs, ys, zs] = True
    if not np.any(mask):
        raise ValueError('Empty target mask')
    if np.any(ct.imageArray[mask] != manifest['water_hu']):
        raise ValueError('Imported CT target does not lie entirely inside water')
    return mask


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ct-dir', required=True, type=Path)
    p.add_argument('--rtplan', type=Path, help='An exported proton PBS DICOM RTPLAN')
    p.add_argument('--out', type=Path, default=Path('open_tps_output'))
    p.add_argument('--nb-primaries', type=int, default=1_000_000)
    p.add_argument('--scanner-folder', type=Path, help='Override default OpenTPS scanner calibration')
    p.add_argument('--bdl-file', type=Path, help='Override default MCsquare commissioned BDL')
    p.add_argument('--water-material-id', type=int,
                   help='Known water material ID in the MCsquare Materials/list.dat file')
    p.add_argument('--skip-calibration-check', action='store_true',
                   help='Only for debugging; DOES NOT verify HU/material/RSP')
    p.add_argument('--gantry-angle', type=float, default=0.,
                   help='Demo-plan gantry angle in degrees (only when --rtplan is absent)')
    p.add_argument('--couch-angle', type=float, default=0.,
                   help='Demo-plan couch angle in degrees')
    p.add_argument('--spot-spacing-mm', type=float, default=5.0)
    p.add_argument('--layer-spacing-mm', type=float, default=5.0)
    p.add_argument('--target-margin-mm', type=float, default=3.0)
    args = p.parse_args()
    if args.nb_primaries < 1 or args.spot_spacing_mm <= 0 or args.layer_spacing_mm <= 0 \
            or args.target_margin_mm < 0:
        p.error('Primaries/spot spacing/layer spacing must be positive; margin >= 0')
    manifest = json.loads((args.ct_dir / 'phantom_geometry.json').read_text())
    ct_files = sorted(args.ct_dir.glob('CT_*.dcm'))
    if len(ct_files) != int(manifest['number_of_slices']):
        raise ValueError('Wrong number of CT slices in CT folder')

    # Delay OpenTPS imports so --help and phantom-generation work without it.
    from opentps.core.io.dicomIO import readDicomCT, readDicomPlan
    from opentps.core.data.images import ROIMask
    from opentps.core.data.plan import ProtonPlanDesign, ProtonPlan
    from opentps.core.io import mcsquareIO
    from opentps.core.io.scannerReader import readScanner
    from opentps.core.processing.doseCalculation.doseCalculationConfig import DoseCalculationConfig
    from opentps.core.processing.doseCalculation.protons.mcsquareDoseCalculator import MCsquareDoseCalculator

    config = DoseCalculationConfig()
    scanner = args.scanner_folder or Path(config.scannerFolder)
    bdl = args.bdl_file or Path(config.bdlFile)
    if not scanner.is_dir() or not bdl.is_file():
        raise FileNotFoundError(f'Scanner calibration directory or BDL not found: {scanner} / {bdl}')
    if not args.skip_calibration_check:
        from check_calibration import audit
        audit(args.ct_dir, scanner, water_material_id=args.water_material_id,
              optional_rsp_check=True)
    else:
        warnings.warn('CALIBRATION CHECK SKIPPED: dose calculation NOT material-validated',
                      RuntimeWarning)

    ct = readDicomCT([str(f) for f in ct_files])
    if str(ct.frameOfReferenceUID) != manifest['frame_of_reference_uid']:
        raise RuntimeError('OpenTPS-imported CT FrameOfReferenceUID mismatch')
    if not np.allclose(np.asarray(ct.origin), manifest['first_pixel_center_lps_mm'],
                       rtol=0, atol=1e-4):
        raise RuntimeError(f'Incorrect CT origin imported: {ct.origin}')
    if not np.allclose(np.asarray(ct.spacing), manifest['spacing_mm_xyz'],
                       rtol=0, atol=1e-5):
        raise RuntimeError(f'Incorrect CT voxel spacing imported: {ct.spacing}')
    if tuple(ct.imageArray.shape) != tuple(manifest['ct_shape_xyz']):
        raise RuntimeError(f'Incorrect CT axes/order: {ct.imageArray.shape}')
    mask = _target_mask(manifest, ct)

    calibration = readScanner(str(scanner))
    calc = MCsquareDoseCalculator()
    calc.ctCalibration = calibration
    calc.beamModel = mcsquareIO.readBDL(str(bdl))
    calc.nbPrimaries = int(args.nb_primaries)

    if args.rtplan:
        from generate_water_phantom import _plan_geometry
        plan_iso, metadata = _plan_geometry(args.rtplan)
        iso = np.asarray(manifest['isocenter_lps_mm'], dtype=float)
        if not np.allclose(plan_iso, iso, rtol=0, atol=1e-3):
            raise RuntimeError('External plan isocenter and phantom isocenter disagree')
        if metadata['frame_uid'] != manifest['frame_of_reference_uid']:
            raise RuntimeError('External plan FrameOfReferenceUID and CT disagree')
        if metadata['patient_id'] != manifest['patient_id']:
            raise RuntimeError('External plan PatientID and phantom disagree')
        if metadata['study_uid'] != manifest['study_instance_uid']:
            raise RuntimeError('External plan StudyInstanceUID and phantom disagree')
        raw = pydicom.dcmread(str(args.rtplan), stop_before_pixels=True)
        if not hasattr(raw, 'IonBeamSequence'):
            raise TypeError('External RTPLAN must be a proton ion plan, not a photon plan')
        all_species = {str(getattr(b, 'RadiationType', '')) for b in raw.IonBeamSequence}
        if all_species != {'PROTON'}:
            raise ValueError(f'MCsquare example expects PROTON beams; found {all_species}')
        if metadata['referenced_rs_uids'] and \
                manifest['rtstruct_sop_instance_uid'] not in metadata['referenced_rs_uids']:
            warnings.warn('RTPLAN references its original RTSTRUCT, NOT the new phantom '
                          'RTSTRUCT. This DICOM set is not automatically a transferable '
                          'plan in RayStation.', RuntimeWarning)
        for beam in raw.IonBeamSequence:
            for tag in ('RangeShifterSequence', 'RangeModulatorSequence',
                        'LateralSpreadingDeviceSequence', 'BlockSequence',
                        'CompensatorSequence'):
                if getattr(beam, tag, None):
                    warnings.warn(f'Plan beam contains {tag}. Confirm MCsquare beam model '
                                  'represents accessory physics; this code does not.', RuntimeWarning)
        plan = readDicomPlan(str(args.rtplan))
        if not isinstance(plan, ProtonPlan):
            raise TypeError('OpenTPS did not import a ProtonPlan from this DICOM RTPLAN')
        plan_mode = 'imported_proton_rtplan'
    else:
        target_roi = ROIMask(imageArray=mask, name='Target', origin=ct.origin,
                             spacing=ct.spacing, patient=ct.patient)
        target_roi.color = (255, 80, 60)
        designer = ProtonPlanDesign()
        designer.ct = ct
        designer.targetMask = target_roi
        designer.gantryAngles = [float(args.gantry_angle)]
        designer.couchAngles = [float(args.couch_angle)]
        designer.beamNames = ['Conventional_PBS_Demo']
        designer.calibration = calibration
        designer.spotSpacing = float(args.spot_spacing_mm)
        designer.layerSpacing = float(args.layer_spacing_mm)
        designer.targetMargin = float(args.target_margin_mm)
        plan = designer.buildPlan()
        plan.PlanName = 'WATER_PHANTOM_RESEARCH_DEMO'
        plan_mode = 'conventional_multilayer_pbs_demo'

    dose = calc.computeDose(ct, plan)
    args.out.mkdir(parents=True, exist_ok=True)
    npz = args.out / 'dose_result.npz'
    np.savez_compressed(
        npz,
        dose=np.asarray(dose.imageArray),
        dose_origin_lps_mm=np.asarray(dose.origin),
        dose_spacing_mm=np.asarray(dose.spacing),
        ct_origin_lps_mm=np.asarray(ct.origin),
        ct_spacing_mm=np.asarray(ct.spacing),
        ct_hu=np.asarray(ct.imageArray),
    )
    (args.out / 'dose_run.json').write_text(json.dumps({
        'plan_mode': plan_mode,
        'imported_rtplan_path': str(args.rtplan) if args.rtplan else None,
        'phantom_manifest': str((args.ct_dir / 'phantom_geometry.json').resolve()),
        'scanner_folder': str(scanner.resolve()),
        'bdl_file': str(bdl.resolve()),
        'nb_primaries': args.nb_primaries,
        'dose_npz': npz.name,
        'dose_units_warning': 'Verify OpenTPS dose normalization and units independently',
        'flash_warning': 'STATIC dose only; no FLASH dose-rate or upstream ridge-filter physics',
    }, indent=2) + '\n')
    print('Computed static proton dose:', npz.resolve())
    print(f'Dose min/max: {np.min(dose.imageArray):.5g}, {np.max(dose.imageArray):.5g}')
    print('NOT MODELED: FLASH dose rate, filter geometry, beamline transmissions, '
          'physical delivery times or radiation biology.')


if __name__ == '__main__':
    main()
