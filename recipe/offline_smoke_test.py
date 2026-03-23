from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.io import fits
from astropy_healpix import HEALPix

from gwtc_analysis.gw_stat import add_localization_area_from_directory
from gwtc_analysis.pe_data_release import PESkyMap
from gwtc_analysis.read_skymap import (
    credible_level_at_radec_percent,
    load_skymap,
    plot_skymap_with_ra_dec,
)


matplotlib.use("Agg", force=True)


@contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def write_test_skymap(path: Path) -> np.ndarray:
    prob = np.full(12, 0.5 / 11.0, dtype=np.float64)
    prob[0] = 0.5

    header = fits.Header()
    header["PIXTYPE"] = "HEALPIX"
    header["ORDERING"] = "NESTED"
    header["NSIDE"] = 1
    header["INDXSCHM"] = "IMPLICIT"

    fits.PrimaryHDU(data=prob, header=header).writeto(path)
    return prob


def expected_credible_area_deg2(prob: np.ndarray, percent: float) -> float:
    ordered = np.sort(np.asarray(prob, dtype=float))[::-1]
    credible = 100.0 * np.cumsum(ordered)
    pixel_area = 41252.96 / float(len(prob))
    return float((credible <= float(percent)).sum() * pixel_area)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        skymap_path = tmpdir / "GW000001_000001-Mixed_test.fits"
        prob = write_test_skymap(skymap_path)

        hp = HEALPix(nside=1, order="nested")
        lon, lat = hp.healpix_to_lonlat(np.array([0]))
        ra_deg = float(lon.to_value(u.deg)[0])
        dec_deg = float(lat.to_value(u.deg)[0])

        skymap = load_skymap(skymap_path)
        cls = credible_level_at_radec_percent(skymap_path, ra_deg, dec_deg)
        if not np.isclose(cls, 50.0, atol=1e-8):
            raise AssertionError(f"unexpected credible level at max-probability pixel: {cls}")

        expected_a90 = expected_credible_area_deg2(prob, 90.0)
        area_a90 = skymap.credible_area_deg2(90.0)
        if not np.isclose(area_a90, expected_a90, atol=1e-8):
            raise AssertionError(f"unexpected A90 from local skymap reader: {area_a90} != {expected_a90}")

        with pushd(tmpdir):
            plot_path = Path(plot_skymap_with_ra_dec(skymap_path, "GW000001_000001", ra_deg, dec_deg, "grey"))
            if not plot_path.exists():
                raise AssertionError(f"search_skymaps plot not created: {plot_path}")

            pe_plot = tmpdir / "pe_skymap.png"
            fig, ax = PESkyMap(prob, meta_data={"nest": True, "distmean": 100.0, "diststd": 10.0}).plot(
                contour=[50, 90]
            )
            fig.savefig(pe_plot, dpi=100)
            fig.clf()
            if not pe_plot.exists():
                raise AssertionError(f"PE skymap plot not created: {pe_plot}")
            if ax is None:
                raise AssertionError("PE skymap plot did not return axes")

        df = pd.DataFrame([{"event_id": "GW000001_000001"}])
        out = add_localization_area_from_directory(df, skymap_dir=tmpdir, cred=0.9, progress=False, verbose=False)
        value = float(out["A90_deg2"].iloc[0])
        if not np.isclose(value, expected_a90, atol=1e-8):
            raise AssertionError(f"unexpected A90 from directory helper: {value} != {expected_a90}")

    print("offline smoke test passed")


if __name__ == "__main__":
    main()
