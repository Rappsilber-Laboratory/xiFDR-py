import logging
import numpy as np
import polars as pl
import pytest

from xifdr.fdr import single_fdr, full_fdr
from xifdr.boosting import boost

@pytest.mark.slow
def test_boosting():
    samples = pl.read_parquet('tests/fixtures/sample_data.parquet')
    fdrs = boost(
        samples,
        csm_fdr=(0, 0.2),
        link_fdr=(0.05, 0.05),
        ppi_fdr=(0.05, 0.05),
        points=3,
        n_jobs=3
    )
    print(fdrs)

def test_column_boost():
    samples = pl.read_parquet('tests/fixtures/sample_data.parquet')
    samples = samples.with_columns(coverage=pl.col('coverage_p1')+pl.col('coverage_p2'))
    cutoffs = boost(
        samples,
        csm_fdr=(0, 0.2),
        link_fdr=(0.05, 0.05),
        ppi_fdr=(0.05, 0.05),
        boost_cols=['coverage'],
        neg_boost_cols=['charge'],
        points=3,
        n_jobs=3
    )
    print(cutoffs)
    pass

def test_boost_startpoints():
    samples = pl.read_parquet('tests/fixtures/sample_data.parquet')
    samples = samples.with_columns(coverage=pl.col('coverage_p1')+pl.col('coverage_p2'))
    cutoffs = boost(
        samples,
        csm_fdr=(0, 0.2),
        link_fdr=(0.05, 0.05),
        ppi_fdr=(0.05, 0.05),
        start_points=(0.05, 0.10, 0.30, 0.05, 0.05),
        points=3,
        n_jobs=3
    )
    print(cutoffs)
    pass
