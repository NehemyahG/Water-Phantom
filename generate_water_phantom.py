"""Create synthetic CT + RTSTRUCT (External and Target) for proton research.

Default geometry: 10 x 10 cm transverse cross-section, 20 cm depth
along DICOM LPS +Y (anterior-to-posterior). A 5 cm air border encloses
all six faces. Coordinates are millimeters in the DICOM LPS system.

This creates *image/structure* objects, not a machine-deliverable RTPLAN.
There is no FLASH, ridge-filter, or dose-rate model in these DICOM objects.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import (
    CTImageStorage, ExplicitVRLittleEndian, RTStructureSetStorage, generate_uid,
)


AXES = 'xyz'


def _count(length_mm: float, spacing_mm: float, label: str) -> int:
    n = round(length_mm / spacing_mm)
    if n < 1 or not np.isclose(n * spacing_mm, length_mm, rtol=0, atol=1e-6):
        raise ValueError(f'{label}: {length_mm:g} mm must be a positive exact multiple '
                         f'of voxel spacing {spacing_mm:g} mm')
    return n


def _box_slices(grid_n: tuple[int, int, int], box_n: tuple[int, int, int],
                shift_mm: np.ndarray, spacing: np.ndarray, label: str
                ) -> tuple[slice, slice, slice]:
    out = []
    for axis, n_fov, n_box, offset, step in zip(AXES, grid_n, box_n, shift_mm, spacing):
        shift_vox = int(round(offset / step))
        if not np.isclose(shift_vox * step, offset, rtol=0, atol=1e-6):
            raise ValueError(f'{label}: offset in {axis} must be an integer multiple '
                             'of voxel spacing for exact voxel-edge alignment')
        if (n_fov - n_box) % 2:
            raise ValueError(f'{label}: FOV minus ROI must be even in voxels along {axis}. '
                             'Adjust the FOV, ROI size or spacing.')
        begin = (n_fov - n_box) // 2 + shift_vox
        if begin < 0 or begin + n_box > n_fov:
            raise ValueError(f'{label}: {axis} extent exceeds the CT field of view')
        out.append(slice(begin, begin + n_box))
    return tuple(out)


def _slices_inside(inner: tuple[slice, ...], outer: tuple[slice, ...]) -> bool:
    return all(o.start <= i.start and i.stop <= o.stop
               for i, o in zip(inner, outer))


def _bounds_from_slices(box: tuple[slice, ...], origin: np.ndarray,
                        spacing: np.ndarray) -> list[list[float]]:
    return [[float(origin[i] + (s.start - .5) * spacing[i]),
             float(origin[i] + (s.stop - .5) * spacing[i])]
            for i, s in enumerate(box)]


def _plan_geometry(plan_path: Path) -> tuple[np.ndarray, dict]:
    """Return (one shared isocenter LPS mm, plan patient & study metadata)."""
    plan = pydicom.dcmread(str(plan_path), stop_before_pixels=True)
    if str(getattr(plan, 'Modality', '')) != 'RTPLAN':
        raise ValueError('--rtplan must be a DICOM photon or ion RTPLAN')
    frame = str(getattr(plan, 'FrameOfReferenceUID', ''))
    if not frame:
        raise ValueError('RTPLAN has no FrameOfReferenceUID')
    positions = {str(s.PatientPosition)
                 for s in getattr(plan, 'PatientSetupSequence', [])
                 if hasattr(s, 'PatientPosition')}
    if positions and positions != {'HFS'}:
        raise ValueError(f'Only HFS RTPLAN is supported; found {positions}')
    centers = []
    for beam_seq, control_seq in [('IonBeamSequence', 'IonControlPointSequence'),
                                  ('BeamSequence', 'ControlPointSequence')]:
        for beam in getattr(plan, beam_seq, []):
            for cp in getattr(beam, control_seq, []):
                pos = getattr(cp, 'IsocenterPosition', None)
                if pos is not None and len(pos) == 3:
                    centers.append(np.asarray([float(c) for c in pos]))
    if not centers:
        raise ValueError('No plan IsocenterPosition in ion/photon control points')
    if not all(np.allclose(x, centers[0], rtol=0, atol=1e-3) for x in centers):
        raise ValueError('Multiple plan isocenters found; a single synthetic CT center is ambiguous')
    if not np.all(np.isfinite(centers[0])):
        raise ValueError('Non-finite RTPLAN isocenter')
    old_rs = [str(x.ReferencedSOPInstanceUID)
              for x in getattr(plan, 'ReferencedStructureSetSequence', [])
              if hasattr(x, 'ReferencedSOPInstanceUID')]
    return centers[0], {
        'frame_uid': frame,
        'study_uid': str(plan.StudyInstanceUID),
        'patient_id': str(getattr(plan, 'PatientID', 'WATER_PHANTOM_001')),
        'patient_name': str(getattr(plan, 'PatientName', 'PHANTOM^WATER')),
        'patient_birth_date': str(getattr(plan, 'PatientBirthDate', '')),
        'patient_sex': str(getattr(plan, 'PatientSex', '')),
        'referenced_rs_uids': old_rs,
    }


def _file_dataset(path: Path, sop_class: str, sop_uid: str) -> FileDataset:
    meta = FileMetaDataset()
    meta.FileMetaInformationVersion = b'\x00\x01'
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0' * 128)
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = sop_uid
    ds.SpecificCharacterSet = 'ISO_IR 192'
    return ds


def _patient_study(ds: Dataset, meta: dict, date: str, time: str) -> None:
    ds.PatientName = meta['patient_name']
    ds.PatientID = meta['patient_id']
    ds.PatientBirthDate = meta['patient_birth_date']
    ds.PatientSex = meta['patient_sex']
    ds.StudyInstanceUID = meta['study_uid']
    ds.StudyID = '1'
    ds.AccessionNumber = ''
    ds.ReferringPhysicianName = ''
    ds.StudyDate = date
    ds.StudyTime = time
    ds.StudyDescription = 'RESEARCH ONLY - synthetic water phantom'
    ds.SeriesDate = date
    ds.SeriesTime = time
    ds.Manufacturer = 'SYNTHETIC'
    ds.ManufacturerModelName = 'Python pydicom phantom generator'


def _image_reference(uid: str) -> Dataset:
    ref = Dataset()
    ref.ReferencedSOPClassUID = CTImageStorage
    ref.ReferencedSOPInstanceUID = uid
    return ref


def _write_rtstruct(path: Path, *, meta: dict, date: str, time: str,
                    frame_uid: str, ct_series_uid: str, ct_uids: list[str],
                    ct_origin_z: float, dz: float,
                    roi_specs: list[dict]) -> str:
    """Write closed planar rectangular contours on their corresponding CT slices.

    Each ROI is voxel-edge aligned in x/y and contoured at CT slice-center z.
    ROIs with an even number of slices are naturally centered between CT slices.
    """
    uid = generate_uid()
    rs = _file_dataset(path, RTStructureSetStorage, uid)
    _patient_study(rs, meta, date, time)
    rs.Modality = 'RTSTRUCT'
    rs.SeriesInstanceUID = generate_uid()
    rs.SeriesNumber = 2
    rs.InstanceNumber = 1
    rs.FrameOfReferenceUID = frame_uid
    rs.PositionReferenceIndicator = ''
    rs.ContentDate = date
    rs.ContentTime = time
    rs.StructureSetLabel = 'WATER_PHANTOM'
    rs.StructureSetName = 'SyntheticWaterPhantom'
    rs.StructureSetDescription = 'Synthetic external and target; research, no patient'
    rs.StructureSetDate = date
    rs.StructureSetTime = time
    rs.ApprovalStatus = 'UNAPPROVED'

    referenced_series = Dataset()
    referenced_series.SeriesInstanceUID = ct_series_uid
    referenced_series.ContourImageSequence = [_image_reference(u) for u in ct_uids]
    referenced_study = Dataset()
    # DICOM permits this retired identifier in RTReferencedStudySequence.
    referenced_study.ReferencedSOPClassUID = '1.2.840.10008.3.1.2.3.1'
    referenced_study.ReferencedSOPInstanceUID = meta['study_uid']
    referenced_study.RTReferencedSeriesSequence = [referenced_series]
    reference = Dataset()
    reference.FrameOfReferenceUID = frame_uid
    reference.RTReferencedStudySequence = [referenced_study]
    rs.ReferencedFrameOfReferenceSequence = [reference]

    rs.StructureSetROISequence = []
    rs.ROIContourSequence = []
    rs.RTROIObservationsSequence = []
    for i, spec in enumerate(roi_specs, start=1):
        roi = Dataset()
        roi.ROINumber = i
        roi.ReferencedFrameOfReferenceUID = frame_uid
        roi.ROIName = spec['name']
        roi.ROIDescription = spec['description']
        roi.ROIGenerationAlgorithm = 'AUTOMATIC'
        size_mm = [b[1] - b[0] for b in spec['bounds_lps_mm_xyz']]
        roi.ROIVolume = float(np.prod(size_mm) / 1000)  # mm^3 -> cm^3
        rs.StructureSetROISequence.append(roi)

        contour_roi = Dataset()
        contour_roi.ReferencedROINumber = i
        contour_roi.ROIDisplayColor = spec['color']
        contour_roi.ContourSequence = []
        bx, by, _ = spec['bounds_lps_mm_xyz']
        z0, z1 = spec['voxel_slices_xyz'][2]
        for k in range(z0, z1):
            z = ct_origin_z + k * dz
            points = [[bx[0], by[0], z], [bx[1], by[0], z],
                      [bx[1], by[1], z], [bx[0], by[1], z]]
            contour = Dataset()
            contour.ContourImageSequence = [_image_reference(ct_uids[k])]
            contour.ContourGeometricType = 'CLOSED_PLANAR'
            contour.NumberOfContourPoints = 4
            contour.ContourData = [float(v) for p in points for v in p]
            contour_roi.ContourSequence.append(contour)
        rs.ROIContourSequence.append(contour_roi)

        obs = Dataset()
        obs.ObservationNumber = i
        obs.ReferencedROINumber = i
        obs.RTROIInterpretedType = spec['type']
        obs.ROIInterpreter = ''
        if spec['type'] == 'EXTERNAL':
            # DICOM annotation. *NOT* a validated RayStation or MCsquare
            # material override. Check the actual calculation calibration.
            properties = []
            for pname in ('REL_MASS_DENSITY', 'REL_STOP_RATIO'):
                prop = Dataset()
                prop.ROIPhysicalProperty = pname
                prop.ROIPhysicalPropertyValue = 1.0
                properties.append(prop)
            obs.ROIPhysicalPropertiesSequence = properties
        rs.RTROIObservationsSequence.append(obs)
    rs.save_as(str(path), enforce_file_format=True)
    return uid


def create_phantom(
    out: Path,
    water_mm: tuple[float, float, float] = (100., 200., 100.),
    fov_mm: tuple[float, float, float] = (200., 300., 200.),
    spacing_mm: tuple[float, float, float] = (2., 2., 2.),
    isocenter_mm: tuple[float, float, float] = (0., 0., 0.),
    water_offset_mm: tuple[float, float, float] = (0., 0., 0.),
    target_mm: tuple[float, float, float] = (60., 40., 60.),
    target_offset_mm: tuple[float, float, float] = (0., 0., 0.),
    water_hu: int = 0,
    air_hu: int = -1000,
    rtplan_path: Path | None = None,
) -> dict:
    """Write one axial CT per slice plus RTSTRUCT and geometry manifest.

    Input dimensions are (x, y, z) in LPS, not image row/column order.
    """
    if rtplan_path is not None:
        iso, patient = _plan_geometry(rtplan_path)
    else:
        iso = np.asarray(isocenter_mm, dtype=float)
        patient = {
            'frame_uid': generate_uid(), 'study_uid': generate_uid(),
            'patient_id': 'WATER_PHANTOM_001', 'patient_name': 'PHANTOM^WATER',
            'patient_birth_date': '', 'patient_sex': 'O',
            'referenced_rs_uids': [],
        }
    water = np.asarray(water_mm, dtype=float)
    fov = np.asarray(fov_mm, dtype=float)
    spacing = np.asarray(spacing_mm, dtype=float)
    water_shift = np.asarray(water_offset_mm, dtype=float)
    target = np.asarray(target_mm, dtype=float)
    target_shift = np.asarray(target_offset_mm, dtype=float)
    for label, values in [('isocenter', iso), ('water', water), ('fov', fov),
                          ('spacing', spacing), ('water offset', water_shift),
                          ('target', target), ('target offset', target_shift)]:
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError(f'{label} must have three finite coordinates')
    if np.any(water <= 0) or np.any(fov <= 0) or np.any(spacing <= 0) or np.any(target <= 0):
        raise ValueError('Water, target, FOV and spacing must be positive')
    if np.any(water >= fov):
        raise ValueError('The water phantom must fit inside the CT with surrounding air')
    if np.any(spacing[:2] > 5):
        raise ValueError('In-plane voxel size > 5 mm is not supported for this interoperability demo')
    for label, hu in [('water', water_hu), ('air', air_hu)]:
        if not isinstance(hu, int) or not -32768 <= hu <= 32767:
            raise ValueError(f'{label} HU must be a signed 16-bit integer')
    if air_hu == water_hu:
        raise ValueError('Air and water HU must differ')

    counts = tuple(_count(float(length), float(step), 'FOV')
                   for length, step in zip(fov, spacing))
    wn = tuple(_count(float(length), float(step), 'water')
               for length, step in zip(water, spacing))
    tn = tuple(_count(float(length), float(step), 'target')
               for length, step in zip(target, spacing))
    nx, ny, nz = counts
    if nx > 65535 or ny > 65535:
        raise ValueError('DICOM Rows/Columns must be <= 65535')
    ws = _box_slices(counts, wn, water_shift, spacing, 'Water')
    ts = _box_slices(counts, tn, target_shift, spacing, 'Target')
    if not _slices_inside(ts, ws):
        raise ValueError('Target must be fully contained within the water phantom')
    origin = iso - fov / 2 + spacing / 2  # center of first pixel, not voxel edge
    frame_uid = patient['frame_uid']
    study_uid = patient['study_uid']
    series_uid = generate_uid()
    now = datetime.now()
    date, time = now.strftime('%Y%m%d'), now.strftime('%H%M%S.%f')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'{out} is not empty; select a new output folder')
    out.mkdir(parents=True, exist_ok=True)
    ct_uids = []
    for k in range(nz):
        pixels = np.full((ny, nx), air_hu, dtype='<i2')
        if ws[2].start <= k < ws[2].stop:
            pixels[ws[1], ws[0]] = water_hu
        name = out / f'CT_{k+1:04d}.dcm'
        ds = _file_dataset(name, CTImageStorage, generate_uid())
        ct_uids.append(str(ds.SOPInstanceUID))
        _patient_study(ds, patient, date, time)
        ds.Modality = 'CT'
        ds.ImageType = ['DERIVED', 'SECONDARY', 'AXIAL']
        ds.DerivationDescription = 'Synthetic water phantom: not an X-ray acquisition'
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_uid
        ds.PositionReferenceIndicator = ''
        ds.PatientPosition = 'HFS'
        ds.SeriesNumber = 1
        ds.InstanceNumber = k + 1
        ds.AcquisitionNumber = 1
        ds.AcquisitionDate = date
        ds.AcquisitionTime = time
        ds.ContentDate = date
        ds.ContentTime = time
        ds.SeriesDescription = ('Synthetic water %gx%gx%g mm' % tuple(water))
        ds.KVP = ''  # DICOM Type 2; no actual X-ray spectrum exists
        ds.Rows = ny
        ds.Columns = nx
        ds.PixelSpacing = [float(spacing[1]), float(spacing[0])]
        ds.SliceThickness = float(spacing[2])
        ds.SpacingBetweenSlices = float(spacing[2])
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [float(origin[0]), float(origin[1]),
                                   float(origin[2] + k * spacing[2])]
        ds.SliceLocation = float(origin[2] + k * spacing[2])
        ds.ReconstructionDiameter = float(max(fov[:2]))
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.RescaleIntercept = 0
        ds.RescaleSlope = 1
        ds.RescaleType = 'HU'
        ds.WindowCenter = -500
        ds.WindowWidth = 1200
        ds.PixelData = pixels.tobytes(order='C')
        ds.save_as(str(name), enforce_file_format=True)

    specs = [
        {'name': 'External', 'type': 'EXTERNAL', 'description': 'Water volume; air excluded',
         'color': [0, 150, 255], 'voxel_slices_xyz': [[s.start, s.stop] for s in ws],
         'bounds_lps_mm_xyz': _bounds_from_slices(ws, origin, spacing)},
        {'name': 'Target', 'type': 'PTV', 'description': 'Synthetic planning target in water',
         'color': [255, 80, 60], 'voxel_slices_xyz': [[s.start, s.stop] for s in ts],
         'bounds_lps_mm_xyz': _bounds_from_slices(ts, origin, spacing)},
    ]
    rs_path = out / 'RS_WATER_PHANTOM.dcm'
    rs_uid = _write_rtstruct(rs_path, meta=patient, date=date, time=time,
                             frame_uid=frame_uid, ct_series_uid=series_uid,
                             ct_uids=ct_uids, ct_origin_z=float(origin[2]),
                             dz=float(spacing[2]), roi_specs=specs)

    info = {
        'description': 'Synthetic water CT/RTSTRUCT for research only',
        'ct_directory': str(out.resolve()),
        'number_of_slices': nz,
        'ct_shape_xyz': list(counts),
        'water_voxels_xyz': list(wn),
        'water_size_mm_xyz': water.tolist(),
        'water_offset_mm_xyz': water_shift.tolist(),
        'target_size_mm_xyz': target.tolist(),
        'target_offset_mm_xyz': target_shift.tolist(),
        'fov_size_mm_xyz': fov.tolist(),
        'spacing_mm_xyz': spacing.tolist(),
        'isocenter_lps_mm': iso.tolist(),
        'first_pixel_center_lps_mm': origin.tolist(),
        'image_orientation_patient': [1, 0, 0, 0, 1, 0],
        'pixel_spacing_row_col_mm': [float(spacing[1]), float(spacing[0])],
        'water_hu': water_hu,
        'air_hu': air_hu,
        'patient_id': patient['patient_id'],
        'study_instance_uid': study_uid,
        'series_instance_uid': series_uid,
        'frame_of_reference_uid': frame_uid,
        'ct_sop_instance_uids': ct_uids,
        'rtstruct_file': rs_path.name,
        'rtstruct_sop_instance_uid': rs_uid,
        'roi_specs': specs,
        'rtplan_source': str(rtplan_path.resolve()) if rtplan_path else None,
        'rtplan_original_referenced_rtstruct_uids': patient['referenced_rs_uids'],
        'calibration_verified': False,
        'note': ('External contour has DICOM relative density and stopping-power annotations '
                 'of 1.0, but these DO NOT guarantee TPS/material assignment. '
                 'An external RTPLAN is NOT automatically relinked to the new RTSTRUCT.'),
    }
    (out / 'phantom_geometry.json').write_text(json.dumps(info, indent=2) + '\n')
    return info


def _expand_dimensions(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if args.water_mm is not None:
        water = np.asarray(args.water_mm, dtype=float)
    else:
        cross = np.asarray(args.cross_section_mm, dtype=float)
        water = np.zeros(3, dtype=float)
        other_axes = [a for a in range(3) if a != AXES.index(args.depth_axis)]
        water[other_axes] = cross
        water[AXES.index(args.depth_axis)] = args.depth_mm
    if args.fov_mm is not None:
        fov = np.asarray(args.fov_mm, dtype=float)
    else:
        fov = water + 2 * args.air_margin_mm
    return water, fov


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=Path('water_phantom_CT'))
    ap.add_argument('--cross-section-mm', nargs=2, type=float, default=[100., 100.],
                    metavar=('WIDTH', 'HEIGHT'), help='Perpendicular dimensions; default 100 100')
    ap.add_argument('--depth-mm', type=float, default=200., help='Water thickness along depth axis')
    ap.add_argument('--depth-axis', choices=list(AXES), default='y',
                    help='LPS depth axis: y is the default AP-style test geometry')
    ap.add_argument('--water-mm', nargs=3, type=float, metavar=('X', 'Y', 'Z'),
                    help='Override cross-section/depth with explicit LPS x,y,z dimensions')
    ap.add_argument('--air-margin-mm', type=float, default=50.,
                    help='Air border on each face in mm, unless --fov-mm specified')
    ap.add_argument('--fov-mm', nargs=3, type=float, metavar=('X', 'Y', 'Z'))
    ap.add_argument('--spacing-mm', nargs=3, type=float, default=[2., 2., 2.],
                    metavar=('DX', 'DY', 'DZ'))
    ap.add_argument('--isocenter-mm', nargs=3, type=float, default=None,
                    metavar=('X', 'Y', 'Z'))
    ap.add_argument('--water-offset-mm', nargs=3, type=float, default=[0., 0., 0.],
                    metavar=('X', 'Y', 'Z'))
    ap.add_argument('--target-mm', nargs=3, type=float, default=[60., 40., 60.],
                    metavar=('X', 'Y', 'Z'))
    ap.add_argument('--target-offset-mm', nargs=3, type=float, default=[0., 0., 0.],
                    metavar=('X', 'Y', 'Z'))
    ap.add_argument('--rtplan', type=Path,
                    help='Adopt FrameOfReferenceUID and single isocenter from exported RTPLAN')
    ap.add_argument('--water-hu', type=int, default=0)
    ap.add_argument('--air-hu', type=int, default=-1000)
    args = ap.parse_args()
    if args.rtplan and args.isocenter_mm is not None:
        ap.error('--rtplan and --isocenter-mm are mutually exclusive')
    water, fov = _expand_dimensions(args)
    m = create_phantom(
        out=args.out, water_mm=tuple(water), fov_mm=tuple(fov),
        spacing_mm=tuple(args.spacing_mm),
        isocenter_mm=tuple(args.isocenter_mm or [0., 0., 0.]),
        water_offset_mm=tuple(args.water_offset_mm),
        target_mm=tuple(args.target_mm),
        target_offset_mm=tuple(args.target_offset_mm),
        water_hu=args.water_hu, air_hu=args.air_hu,
        rtplan_path=args.rtplan,
    )
    print(f"Created {m['number_of_slices']} DICOM CT slices and {m['rtstruct_file']} in {args.out}")
    print('Water dimensions LPS x,y,z (mm):', m['water_size_mm_xyz'])
    print('DICOM isocenter LPS (mm):', m['isocenter_lps_mm'])
    print('RTSTRUCT references CT SOPInstanceUIDs and frame:', m['frame_of_reference_uid'])
    if args.rtplan and m['rtplan_original_referenced_rtstruct_uids']:
        print('WARNING: Input RTPLAN references a different RTSTRUCT UID. '
              'RayStation plan linking must be handled/replanned in a research database.')
    print('WARNING: Material/RSP is NOT validated until check_calibration.py is run.')


if __name__ == '__main__':
    main()
