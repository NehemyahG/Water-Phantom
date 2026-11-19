"""Audit MCsquare HU-to-mass-density and HU-to-material tables for a phantom.

This is a pre-flight check. Matching 1 g/cm^3 does NOT prove that the
water material composition or energy-dependent stopping-power ratio is
correct. RayStation uses its *own* commissioned calibration/material assignment.

Example:
    python check_calibration.py --ct-dir water_phantom_CT \\
        --scanner-folder /path/to/MCsquare/Scanners/YOUR_SCANNER
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
import hashlib
import json
from pathlib import Path

import numpy as np


def _read_two_column_file(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f'Calibration file not found: {path}')
    data = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.split('#', 1)[0].strip().replace(',', ' ')
        if not stripped:
            continue
        chunks = stripped.split()
        if len(chunks) < 2:
            continue
        try:
            hu = float(chunks[0])
            val = float(chunks[1])
        except ValueError:
            # Allow column-name header lines, but not corrupted numeric rows.
            if chunks[0][0] in '-+0123456789.':
                raise ValueError(f'{path.name}, line {line_number}: invalid numeric row')
            continue
        if not (np.isfinite(hu) and np.isfinite(val)):
            raise ValueError(f'{path.name}, line {line_number}: non-finite values')
        data.append([hu, val])
    if len(data) < 2:
        raise ValueError(f'Need >=2 HU entries in {path}')
    array = np.asarray(data)
    order = np.argsort(array[:, 0])
    array = array[order]
    if np.any(np.diff(array[:, 0]) == 0):
        raise ValueError(f'Duplicate HU breakpoints in {path}')
    return array


def _density_at(hu: int, table: np.ndarray) -> float:
    x, y = table[:, 0], table[:, 1]
    if hu < x[0] or hu > x[-1]:
        raise ValueError(f'HU={hu} outside density table span [{x[0]:g}, {x[-1]:g}]. '
                         'Do not silently extrapolate phantom calibration.')
    return float(np.interp(hu, x, y))


def _material_at(hu: int, table: np.ndarray) -> int:
    idx = bisect_right(table[:, 0].tolist(), hu) - 1
    if idx < 0:
        raise ValueError(f'HU={hu} is below material conversion table minimum')
    mat = float(table[idx, 1])
    if not mat.is_integer():
        raise ValueError('Material index must be an integer in MCsquare table')
    return int(mat)


def audit(ct_dir: Path, scanner_folder: Path, *, mass_density_tolerance: float = .02,
          water_material_id: int | None = None, air_material_id: int | None = None,
          optional_rsp_check: bool = True) -> dict:
    if mass_density_tolerance < 0:
        raise ValueError('Mass density tolerance must not be negative')
    manifest = json.loads((ct_dir / 'phantom_geometry.json').read_text())
    water_hu, air_hu = int(manifest['water_hu']), int(manifest['air_hu'])
    density_file = scanner_folder / 'HU_Density_Conversion.txt'
    material_file = scanner_folder / 'HU_Material_Conversion.txt'
    density_table = _read_two_column_file(density_file)
    material_table = _read_two_column_file(material_file)
    density_water = _density_at(water_hu, density_table)
    density_air = _density_at(air_hu, density_table)
    material_water = _material_at(water_hu, material_table)
    material_air = _material_at(air_hu, material_table)
    density_ok = abs(density_water - 1.0) <= mass_density_tolerance
    material_ok = (water_material_id is None or water_material_id == material_water)
    air_material_ok = (air_material_id is None or air_material_id == material_air)
    air_density_ok = 0.0 <= density_air < 0.05
    result = {
        'scanner_folder': str(scanner_folder.resolve()),
        'HU_Density_Conversion_sha256': hashlib.sha256(density_file.read_bytes()).hexdigest(),
        'HU_Material_Conversion_sha256': hashlib.sha256(material_file.read_bytes()).hexdigest(),
        'water_hu': water_hu,
        'air_hu': air_hu,
        'water_density_g_cm3': density_water,
        'air_density_g_cm3': density_air,
        'water_material_id': material_water,
        'air_material_id': material_air,
        'water_mass_density_within_tolerance': bool(density_ok),
        'air_mass_density_plausible': bool(air_density_ok),
        'material_id_matches_user_expected': bool(material_ok and air_material_ok),
        'water_material_identity_independently_verified': water_material_id is not None,
        'water_rsp': None,
        'water_rsp_within_2pct': None,
        'note': ('Verify material-ID->composition in MCsquare Materials/list.dat; '
                 'RayStation uses a separate calibration. This is not a beam model audit.'),
    }
    if optional_rsp_check:
        try:
            from opentps.core.io.scannerReader import readScanner
        except ImportError:
            print('NOTE: OpenTPS not installed here: RSP conversion could not be checked.')
        else:
            calibration = readScanner(str(scanner_folder))
            method = getattr(calibration, 'convertHU2RSP', None)
            if callable(method):
                value = float(np.asarray(method(float(water_hu))).reshape(-1)[0])
                result['water_rsp'] = value
                result['water_rsp_within_2pct'] = bool(np.isfinite(value) and
                                                         abs(value - 1.0) <= .02)
            else:
                print('NOTE: OpenTPS calibration does not expose convertHU2RSP in this version.')

    print(f'Using MCsquare scanner folder: {scanner_folder}')
    print(f'Water HU {water_hu}: density {density_water:.6g} g/cm^3; '
          f'material ID {material_water}')
    print(f'Air HU {air_hu}: density {density_air:.6g} g/cm^3; '
          f'material ID {material_air}')
    if result['water_rsp'] is not None:
        print(f'Water RSP (OpenTPS calibration): {result["water_rsp"]:.6g}')
    if water_material_id is None:
        print('UNVERIFIED: water material identity; compare material ID with '
              'MCsquare Materials/list.dat or use --water-material-id.')
    if not density_ok:
        print(f'FAIL: water mass density differs from 1.0 by more than '
              f'{mass_density_tolerance:.3g} g/cm^3')
    if not air_density_ok:
        print('FAIL: air mass density outside plausible <0.05 g/cm^3 range')
    if not material_ok or not air_material_ok:
        print('FAIL: calibration material-ID mismatch with expected material ID')
    if result['water_rsp_within_2pct'] is False:
        print('FAIL: water RSP differs from 1 by >2%; calibration needs review')
    good = (density_ok and air_density_ok and material_ok and air_material_ok and
            result['water_rsp_within_2pct'] is not False)
    result['preflight_passed'] = bool(good)
    (ct_dir / 'calibration_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    if not good:
        raise ValueError('MCsquare HU conversion preflight failed. '
                         'See calibration_audit.json; do not compute dose yet.')
    print('PASS: numeric MCsquare density checks. '
          'Material composition, calibration and beam model require independent QA.')
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ct-dir', type=Path, required=True)
    ap.add_argument('--scanner-folder', type=Path, default=None,
                    help='MCsquare Scanner folder; if omitted, uses OpenTPS DoseCalculationConfig')
    ap.add_argument('--mass-density-tol', type=float, default=.02)
    ap.add_argument('--water-material-id', type=int,
                    help='Known water ID from local MCsquare Materials/list.dat')
    ap.add_argument('--air-material-id', type=int,
                    help='Known air material ID from local MCsquare Materials/list.dat')
    ap.add_argument('--no-rsp-check', action='store_true',
                    help='Skip optional OpenTPS-calculated RSP check')
    args = ap.parse_args()
    scanner_folder = args.scanner_folder
    if scanner_folder is None:
        from opentps.core.processing.doseCalculation.doseCalculationConfig import DoseCalculationConfig
        scanner_folder = Path(DoseCalculationConfig().scannerFolder)
    audit(args.ct_dir, scanner_folder, mass_density_tolerance=args.mass_density_tol,
          water_material_id=args.water_material_id,
          air_material_id=args.air_material_id,
          optional_rsp_check=not args.no_rsp_check)


if __name__ == '__main__':
    main()
