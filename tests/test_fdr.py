import logging
import numpy as np
import polars as pl

from xifdr.fdr import single_fdr, full_fdr
from xifdr.boosting import boost

logging.basicConfig(level=logging.DEBUG)

def test_single_fdr_large():
    tt_arr = np.arange(10_000).astype(float)
    td_arr = np.arange(10_000).astype(float)
    dt_arr = np.arange(10_000).astype(float)
    dd_arr = np.arange(10_000).astype(float)

    # Shift TD and DD to worse scores
    td_offset = 1000
    dd_offset = 1500
    td_arr -= td_offset
    dt_arr -= td_offset
    dd_arr -= dd_offset

    # Avoid one-off on same score
    tt_arr = tt_arr - .9
    td_arr = td_arr - .6
    dt_arr = dt_arr - .3

    tt_df = pl.DataFrame({
        'score': tt_arr
    })
    td_df = pl.DataFrame({
        'score': np.concat([td_arr, dt_arr])
    })
    dd_df = pl.DataFrame({
        'score': dd_arr
    })

    tt_df = tt_df.with_columns(
        pl.lit(True).alias('TT'),
        pl.lit(False).alias('TD'),
        pl.lit(False).alias('DD'),
    )
    td_df = td_df.with_columns(
        pl.lit(False).alias('TT'),
        pl.lit(True).alias('TD'),
        pl.lit(False).alias('DD'),
    )
    dd_df = dd_df.with_columns(
        pl.lit(False).alias('TT'),
        pl.lit(False).alias('TD'),
        pl.lit(True).alias('DD'),
    )

    df = pl.concat([
        tt_df,
        td_df,
        dd_df
    ])

    df = df.with_columns(
        single_fdr(df)
    )

    df = df.sort('score', descending=True)

    # Check that FDR is 0 before first TD
    assert(
        all(
            np.isclose(df['fdr'].to_numpy()[:td_offset], 0)
        )
    )

    # Check that FDR is 0.5 before first DD
    # Should've seen 2*td_offset TT and 2*td_offset TD
    dd_exp = 500  # Get data up to first 500 DD
    td_exp = 2000  # Before first DD there are 1000 TD and also 1000 TD parallel to DDs
    tt_exp = 2000  # 1000 TTs before first TD and 1000 TTs parallel to TDs
    fdr_exp = (td_exp - dd_exp) / tt_exp
    n_samples = dd_exp + td_exp + tt_exp
    assert(
        np.isclose(
            df['fdr'].to_numpy()[n_samples],
            fdr_exp,
            rtol=0.001)
    )

    # Last should be 100% FDR
    assert (
        np.isclose(
            df['fdr'].to_numpy()[-1],
            1,
            rtol=0.001)
    )


def test_single_fdr_monotone():
    matches = [
        [100, True, False, False, 0.0, 0.0]
    ]
    matches = matches*10
    matches += [
        # 2TD + 1DD
        [99, False, True, False, 0.1, 0.0],
        [99, False, False, True, 0.0, 0.0],
        [98, False, True, False, 0.1, 0.1],
        # 2TD
        [97, False, True, False, 0.2, 0.2],
        [96, False, True, False, 0.3, 0.2],
        # 1DD
        [95, False, False, True, 0.2, 0.2],
        # 3 TD
        [94, False, True, False, 0.3, 0.3],
        [93, False, True, False, 0.4, 0.4],
        [92, False, True, False, 0.5, 0.5],
    ]

    df = pl.DataFrame(
        matches,
        schema=['score', 'TT', 'TD', 'DD', 'exp_raw_fdr', 'exp_fdr'],
        orient = "row"
    ).sample(len(matches), shuffle=True, seed=0)

    df = df.with_columns(
        single_fdr(df)
    )

    df = df.sort('score')

    assert(all(df['fdr'] == df['exp_fdr']))


def test_full_fdr():
    samples = pl.read_parquet('../samples_data.parquet')
    #x = full_fdr(samples)
    #pass
    x = full_fdr(
        samples,
        psm_fdr=0.5,
        pep_fdr=0.5,
        prot_fdr=0.3,
        link_fdr=0.05,
        ppi_fdr=0.05
    )
    pass


def test_boosting():
    samples = pl.read_parquet('../samples_data.parquet')
    fdrs = boost(
        samples,
        link_fdr=(0, 0.05),
        ppi_fdr=(0, 0.05),
        points=5,
        n_jobs=4,
        method="independent_grid"
    )
    print(fdrs)
    assert(False)
