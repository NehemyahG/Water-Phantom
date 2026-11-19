"""Export a central-axis depth-dose CSV from an OpenTPS static dose result.

Use for comparing candidate SOBP/depth-dose curves. Assumes a BEAM ALIGNED
with the chosen Cartesian phantom depth axis; not valid for oblique fields.
Does NOT calculate dose rate or assess biological FLASH effects.

Example:
    python analyze_depth_dose.py --ct-dir water_phantom_CT \\
        --dose-npz open_tps_output/dose_result.npz --axis y --entry negative
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def calculate(ct_dir: Path, dose_npz: Path, *, axis: str = 'y',
              entry: str = 'negative', radius_mm: float = 5.,
              csv_path: Path | None = None,
              plateau_mm: tuple[float, float] | None = None) -> dict:
    if axis not in 'xyz' or entry not in ('negative', 'positive'):
        raise ValueError('Axis must be x/y/z, entry negative/positive')
    if not np.isfinite(radius_mm) or radius_mm <= 0:
        raise ValueError('radius_mm must be positive and finite')
    m = json.loads((ct_dir / 'phantom_geometry.json').read_text())
    water = next(r for r in m['roi_specs'] if r['name'] == 'External')
    bbox = np.asarray(water['bounds_lps_mm_xyz'], dtype=float)
    center = np.asarray(m['isocenter_lps_mm'], dtype=float) + \
        np.asarray(m['water_offset_mm_xyz'], dtype=float)
    with np.load(dose_npz) as d:
        dose = np.asarray(d['dose'], dtype=float)
        origin = np.asarray(d['dose_origin_lps_mm'], dtype=float)
        spacing = np.asarray(d['dose_spacing_mm'], dtype=float)
    if dose.ndim != 3 or not np.all(np.isfinite(dose)):
        raise ValueError('Dose array must be a finite 3D x,y,z array')
    if np.any(spacing <= 0):
        raise ValueError('Dose voxel spacing must be positive')
    coords = [origin[a] + np.arange(dose.shape[a]) * spacing[a] for a in range(3)]
    ia = 'xyz'.index(axis)
    others = [a for a in range(3) if a != ia]
    aa, bb = others
    ca, cb = np.meshgrid(coords[aa], coords[bb], indexing='ij')
    disk = (ca - center[aa]) ** 2 + (cb - center[bb]) ** 2 <= radius_mm ** 2
    disk &= ((ca >= bbox[aa, 0]) & (ca <= bbox[aa, 1]) &
             (cb >= bbox[bb, 0]) & (cb <= bbox[bb, 1]))
    if not np.any(disk):
        raise ValueError('Central dose sampling disk contains no dose grid points')
    doses, depths = [], []
    for i, pos in enumerate(coords[ia]):
        if not bbox[ia, 0] <= pos <= bbox[ia, 1]:
            continue
        plane = np.take(dose, i, axis=ia)
        # np.take() leaves the other two axes in original relative order.
        doses.append(float(np.mean(plane[disk])))
        depth = (pos - bbox[ia, 0]) if entry == 'negative' else (bbox[ia, 1] - pos)
        depths.append(float(depth))
    if not depths:
        raise ValueError('No dose samples intersect the water phantom along the selected axis')
    ind = np.argsort(depths)
    depths = np.asarray(depths)[ind]
    doses = np.asarray(doses)[ind]
    peak = float(np.max(doses))
    normalized = doses / peak if peak > 0 else np.zeros_like(doses)
    csv_path = csv_path or dose_npz.parent / 'central_axis_depth_dose.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['depth_from_water_entry_mm', 'mean_dose_unverified_units',
                         'dose_relative_to_peak'])
        writer.writerows(zip(depths, doses, normalized))
    result = {'csv': str(csv_path), 'axis': axis, 'entry': entry,
              'disk_radius_mm': radius_mm, 'samples': len(depths),
              'max_dose_unverified_units': peak}
    if plateau_mm is not None:
        lo, hi = plateau_mm
        if not 0 <= lo < hi or not np.all(np.isfinite([lo, hi])):
            raise ValueError('Plateau boundaries must satisfy 0 <= start < end')
        plateau = (depths >= lo) & (depths <= hi)
        if np.count_nonzero(plateau) < 2:
            raise ValueError('SOBP plateau has fewer than two sampled depths')
        segment = doses[plateau]
        mean = float(segment.mean())
        flatness = float((np.max(segment) - np.min(segment)) / mean) if mean > 0 else None
        result['plateau_depth_mm'] = [lo, hi]
        result['plateau_peak_to_valley_over_mean'] = flatness
    print(f'Wrote {len(depths)} depth-dose samples to {csv_path}')
    if 'plateau_peak_to_valley_over_mean' in result:
        print('Candidate plateau (max-min)/mean:', result['plateau_peak_to_valley_over_mean'])
    print('This is a static dose profile, not a FLASH dose-rate evaluation.')
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ct-dir', required=True, type=Path)
    p.add_argument('--dose-npz', required=True, type=Path)
    p.add_argument('--axis', default='y', choices=list('xyz'))
    p.add_argument('--entry', default='negative', choices=['negative', 'positive'])
    p.add_argument('--radius-mm', type=float, default=5.)
    p.add_argument('--plateau-mm', nargs=2, type=float, metavar=('START', 'END'))
    p.add_argument('--csv', type=Path)
    args = p.parse_args()
    calculate(args.ct_dir, args.dose_npz, axis=args.axis, entry=args.entry,
              radius_mm=args.radius_mm, csv_path=args.csv,
              plateau_mm=tuple(args.plateau_mm) if args.plateau_mm else None)


if __name__ == '__main__':
    main()
