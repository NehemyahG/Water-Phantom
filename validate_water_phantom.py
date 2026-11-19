"""Validate CT geometry, CT voxels, RTSTRUCT contour geometry/references.

Usage:
    python validate_water_phantom.py --ct-dir water_phantom_CT
    python validate_water_phantom.py --ct-dir aligned_CT --rtplan my_plan.dcm

This tests DICOM contents, not vendor import, HU calibration, beamline physics,
or clinical safety.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pydicom
from pydicom.uid import CTImageStorage, RTStructureSetStorage

from generate_water_phantom import _plan_geometry


def require(test: bool, message: str) -> None:
    if not test:
        raise AssertionError(message)


def _slices(roi: dict) -> tuple[slice, slice, slice]:
    return tuple(slice(int(a), int(b)) for a, b in roi['voxel_slices_xyz'])


def validate(ct_dir: Path, rtplan: Path | None = None) -> None:
    manifest = json.loads((ct_dir / 'phantom_geometry.json').read_text())
    ct_files = sorted(ct_dir.glob('CT_*.dcm'))
    nx, ny, nz = map(int, manifest['ct_shape_xyz'])
    spacing = np.asarray(manifest['spacing_mm_xyz'], dtype=float)
    iso = np.asarray(manifest['isocenter_lps_mm'], dtype=float)
    fov = np.asarray(manifest['fov_size_mm_xyz'], dtype=float)
    origin = iso - fov / 2 + spacing / 2
    require(len(ct_files) == nz, f'Expected {nz} CT slices, found {len(ct_files)}')
    require(np.allclose(origin, manifest['first_pixel_center_lps_mm']),
            'Origin and FOV disagree')

    rois = {r['name']: r for r in manifest['roi_specs']}
    require(set(rois) == {'External', 'Target'}, 'Unexpected ROI names')
    w = _slices(rois['External'])
    t = _slices(rois['Target'])
    for ti, wi in zip(t, w):
        require(wi.start <= ti.start < ti.stop <= wi.stop,
                'Target not contained in water')

    expected_ct_uids = manifest['ct_sop_instance_uids']
    require(len(expected_ct_uids) == nz and len(set(expected_ct_uids)) == nz,
            'Bad CT UID list in manifest')
    seen_water = seen_air = 0
    for k, file in enumerate(ct_files):
        ds = pydicom.dcmread(str(file))
        require(ds.Modality == 'CT' and str(ds.SOPClassUID) == str(CTImageStorage),
                f'Not a CT slice: {file.name}')
        require(str(ds.SOPInstanceUID) == expected_ct_uids[k],
                f'Mismatched SOP instance UID in CT slice {k}')
        for attr, key in [('StudyInstanceUID', 'study_instance_uid'),
                          ('SeriesInstanceUID', 'series_instance_uid'),
                          ('FrameOfReferenceUID', 'frame_of_reference_uid'),
                          ('PatientID', 'patient_id')]:
            require(str(getattr(ds, attr)) == manifest[key], f'{attr} differs in {file.name}')
        require(ds.Rows == ny and ds.Columns == nx,
                f'Unexpected rows/columns on slice {k}')
        require(np.allclose(np.asarray(ds.PixelSpacing, float), spacing[[1, 0]]),
                f'PixelSpacing differs in slice {k}')
        require(np.allclose(np.asarray(ds.ImageOrientationPatient, float),
                            [1, 0, 0, 0, 1, 0]), 'Unexpected DICOM direction cosines')
        require(np.allclose(np.asarray(ds.ImagePositionPatient, float),
                            origin + np.asarray([0, 0, k * spacing[2]]),
                            rtol=0, atol=1e-4), f'ImagePositionPatient differs in slice {k}')
        require(np.isclose(float(ds.SliceThickness), spacing[2]) and
                np.isclose(float(ds.SpacingBetweenSlices), spacing[2]),
                'Unexpected slice separation')
        require(int(ds.BitsAllocated) == 16 and int(ds.PixelRepresentation) == 1,
                'Incorrect pixel representation')
        require(float(ds.RescaleSlope) == 1 and float(ds.RescaleIntercept) == 0,
                'Incorrect HU rescale transform')
        arr = np.full((ny, nx), int(manifest['air_hu']), dtype=np.int16)
        if w[2].start <= k < w[2].stop:
            arr[w[1], w[0]] = int(manifest['water_hu'])
        require(np.array_equal(ds.pixel_array, arr), f'Incorrect pixel values in slice {k}')
        seen_water += int(np.count_nonzero(arr == manifest['water_hu']))
        seen_air += int(np.count_nonzero(arr == manifest['air_hu']))

    require(seen_water == np.prod(manifest['water_voxels_xyz']),
            'Total water voxels incorrect')
    require(seen_water + seen_air == nx * ny * nz, 'Unexpected image pixel counts')

    rs_file = ct_dir / manifest['rtstruct_file']
    rs = pydicom.dcmread(str(rs_file))
    require(str(rs.SOPClassUID) == str(RTStructureSetStorage), 'RS file not an RTSTRUCT')
    require(rs.Modality == 'RTSTRUCT', 'RS Modality incorrect')
    require(str(rs.SOPInstanceUID) == manifest['rtstruct_sop_instance_uid'],
            'RTSTRUCT SOPInstanceUID mismatch')
    for attr, key in [('StudyInstanceUID', 'study_instance_uid'),
                      ('FrameOfReferenceUID', 'frame_of_reference_uid'),
                      ('PatientID', 'patient_id')]:
        require(str(getattr(rs, attr)) == manifest[key], f'RTSTRUCT {attr} mismatch')
    require(str(rs.ApprovalStatus) == 'UNAPPROVED', 'Synthetic RTSTRUCT must be unapproved')
    require(len(rs.ReferencedFrameOfReferenceSequence) == 1,
            'Expected one referenced frame of reference')
    ref = rs.ReferencedFrameOfReferenceSequence[0]
    require(str(ref.FrameOfReferenceUID) == manifest['frame_of_reference_uid'],
            'RTSTRUCT referenced FrameOfReferenceUID mismatch')
    require(len(ref.RTReferencedStudySequence) == 1,
            'Expected one RTReferencedStudySequence')
    study = ref.RTReferencedStudySequence[0]
    require(str(study.ReferencedSOPInstanceUID) == manifest['study_instance_uid'],
            'RTSTRUCT references wrong CT StudyInstanceUID')
    require(len(study.RTReferencedSeriesSequence) == 1, 'Expected one referenced series')
    series = study.RTReferencedSeriesSequence[0]
    require(str(series.SeriesInstanceUID) == manifest['series_instance_uid'],
            'RTSTRUCT references wrong CT SeriesInstanceUID')
    require([str(x.ReferencedSOPInstanceUID) for x in series.ContourImageSequence]
            == expected_ct_uids, 'RTSTRUCT CT slice reference list mismatch')
    require(all(str(x.ReferencedSOPClassUID) == str(CTImageStorage)
                for x in series.ContourImageSequence), 'Incorrect image SOP class reference')

    require(len(rs.StructureSetROISequence) == 2 and len(rs.ROIContourSequence) == 2
            and len(rs.RTROIObservationsSequence) == 2, 'Expected exactly two structures')
    for i, name in enumerate(('External', 'Target'), start=1):
        spec = rois[name]
        bbox = np.asarray(spec['bounds_lps_mm_xyz'], float)
        slices = _slices(spec)
        derived_bounds = np.asarray([
            [origin[axis] + (s.start - .5) * spacing[axis],
             origin[axis] + (s.stop - .5) * spacing[axis]]
            for axis, s in enumerate(slices)
        ])
        require(np.allclose(bbox, derived_bounds, rtol=0, atol=1e-5),
                f'{name} contour bounds do not match CT voxel edges')
        roi = rs.StructureSetROISequence[i - 1]
        cont = rs.ROIContourSequence[i - 1]
        obs = rs.RTROIObservationsSequence[i - 1]
        require(str(roi.ROIName) == name and int(roi.ROINumber) == i,
                f'{name} ROI label/number mismatch')
        require(str(roi.ReferencedFrameOfReferenceUID) == manifest['frame_of_reference_uid'],
                f'{name} wrong frame UID')
        require(int(cont.ReferencedROINumber) == i and int(obs.ReferencedROINumber) == i,
                f'{name} ROIContour/Observation numbers disagree')
        require(str(obs.RTROIInterpretedType) == spec['type'],
                f'{name} RTROIInterpretedType mismatch')
        actual_volume_cm3 = float(np.prod(bbox[:, 1] - bbox[:, 0]) / 1000)
        require(np.isclose(float(roi.ROIVolume), actual_volume_cm3),
                f'{name} physical ROI volume inconsistent')
        count = slices[2].stop - slices[2].start
        require(len(cont.ContourSequence) == count,
                f'{name} wrong contour slice count')
        for k, contour in zip(range(slices[2].start, slices[2].stop), cont.ContourSequence):
            require(contour.ContourGeometricType == 'CLOSED_PLANAR'
                    and contour.NumberOfContourPoints == 4, 'Wrong contour geometry')
            require(len(contour.ContourImageSequence) == 1
                    and str(contour.ContourImageSequence[0].ReferencedSOPInstanceUID)
                    == expected_ct_uids[k], f'{name} wrong per-slice reference')
            z = float(origin[2] + k * spacing[2])
            expected = np.asarray([[bbox[0, 0], bbox[1, 0], z],
                                   [bbox[0, 1], bbox[1, 0], z],
                                   [bbox[0, 1], bbox[1, 1], z],
                                   [bbox[0, 0], bbox[1, 1], z]])
            actual = np.asarray(contour.ContourData, float).reshape((-1, 3))
            require(np.allclose(expected, actual, rtol=0, atol=1e-4),
                    f'{name} contour points do not match geometry at slice {k}')
        if name == 'External':
            properties = {str(x.ROIPhysicalProperty): float(x.ROIPhysicalPropertyValue)
                          for x in obs.ROIPhysicalPropertiesSequence}
            require(properties == {'REL_MASS_DENSITY': 1.0, 'REL_STOP_RATIO': 1.0},
                    'Expected research-only relative water property annotations')

    if rtplan:
        plan_iso, plan_data = _plan_geometry(rtplan)
        require(np.allclose(iso, plan_iso, rtol=0, atol=1e-3),
                'RTPLAN and phantom isocenters differ')
        require(plan_data['frame_uid'] == manifest['frame_of_reference_uid'],
                'RTPLAN and phantom frame of reference differ')
        require(plan_data['patient_id'] == manifest['patient_id'],
                'RTPLAN and phantom patient IDs differ')
        require(plan_data['study_uid'] == manifest['study_instance_uid'],
                'RTPLAN and phantom study UIDs differ')
        print('PASS: External RTPLAN patient, study, frame UID and isocenter match.')
        if plan_data['referenced_rs_uids']:
            if manifest['rtstruct_sop_instance_uid'] not in plan_data['referenced_rs_uids']:
                print('WARNING: External RTPLAN still references its ORIGINAL RTSTRUCT. '
                      'It has NOT been relinked to this new one.')

    print(f'PASS: {nz} CT slices, {seen_water:,} water voxels, {seen_air:,} air voxels.')
    print('PASS: DICOM LPS position/orientation, UID chain, and pixel integrity.')
    print('PASS: RTSTRUCT External/Target closed contours and CT SOP references.')
    print('NOT TESTED: TPS import, MCsquare material/RSP conversion, FLASH dose rate.')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ct-dir', type=Path, required=True)
    p.add_argument('--rtplan', type=Path)
    a = p.parse_args()
    validate(a.ct_dir, a.rtplan)


if __name__ == '__main__':
    main()
