from xifdr.fdr import single_grouped_fdr
import logging
import numpy as np
import polars as pl
import pytest

from xifdr.fdr import single_fdr, full_fdr
from xifdr.boosting import boost

logging.basicConfig(level=logging.DEBUG)

from pathlib import Path
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_data.parquet"

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
    fdr_vals = df['fdr'].to_numpy()

    # FDR must be monotonically non-decreasing as score decreases and stay within [0, 1]
    assert np.all(np.diff(fdr_vals) >= -1e-12)
    assert np.all((fdr_vals >= 0.0) & (fdr_vals <= 1.0))

    # Top match is in the tail after the last TD window (5 TDs, 0 DDs, 2 TTs inside + 1000 tail TTs)
    assert np.isclose(fdr_vals[0], 5.0 / (2 + td_offset), rtol=1e-9)

    # At the lowest pure TT before the first TD (idx td_offset - 1), tail local FDRs average cumulatively
    expected_tail_fdr = np.mean(np.minimum(1.0, 5.0 / (2 + np.arange(1, td_offset + 1))))
    assert np.isclose(fdr_vals[td_offset - 1], expected_tail_fdr, rtol=1e-9)

    # Check FDR rises cleanly deeper into the decoy region and stays strictly below 1 at the bottom
    dd_exp = 500
    td_exp = 2000
    tt_exp = 2000
    n_samples = dd_exp + td_exp + tt_exp
    assert fdr_vals[n_samples] > fdr_vals[td_offset - 1]
    assert fdr_vals[-1] >= fdr_vals[n_samples]
    assert fdr_vals[-1] < 1.0


def test_single_fdr_monotone():
    matches = [
        [100, True, False, False]
    ] * 10
    matches += [
        # 2TD + 1DD
        [99, False, True, False],
        [98, False, False, True],
        [97, False, True, False],
        # 2TD
        [96, False, True, False],
        [95, False, True, False],
        # 1DD
        [94, False, False, True],
        # 3 TD
        [93, False, True, False],
        [92, False, True, False],
        [91, False, True, False],
    ]

    df = pl.DataFrame(
        matches,
        schema=['score', 'TT', 'TD', 'DD'],
        orient="row"
    ).sample(len(matches), shuffle=True, seed=0)

    df = df.with_columns(
        single_fdr(df)
    )

    df_asc = df.sort('score')
    fdr_asc = df_asc['fdr'].to_numpy()

    # Last TD window (scores 93..99) has 5 TDs, 2 DDs, 0 TTs -> TD - DD = 3
    # The 10 tail TTs at score 100 get local pep_k = min(1.0, 3.0 / k) for k=1..10
    tail_peps_asc = np.minimum(1.0, 3.0 / np.arange(1, 11))
    # Cumulative global FDR from highest score (k=10) down to lowest (k=1)
    tail_fdr_desc = np.cumsum(tail_peps_asc[::-1]) / np.arange(1, 11)
    tail_fdr_asc = tail_fdr_desc[::-1]
    # Decoys at scores 91..99 carry the top-10 TT cumulative FDR
    expected_asc = np.concatenate([np.full(9, tail_fdr_asc[0]), tail_fdr_asc])
    expected_asc = np.minimum.accumulate(expected_asc)

    assert np.allclose(fdr_asc, expected_asc, atol=1e-12)
    assert np.all(np.diff(fdr_asc) <= 1e-12)

    # Also test monotonic q-value enforcement when a lower-score DD burst decreases cumulative FDR
    df_burst = pl.DataFrame({
        'score': list(range(1, 13)),
        'TT': [True, True, False, True, True, True, False, True, False, True, True, True],
        'TD': [False, False, True, False, False, False, True, False, True, False, False, False],
        'DD': [False] * 12,
    })
    fdr_burst = single_fdr(df_burst, window_size=2).to_numpy()
    assert np.all(np.diff(fdr_burst) <= 1e-12)
    assert np.isclose(fdr_burst[0], 1541.0 / 3780.0, atol=1e-12)
    assert np.isclose(fdr_burst[2], 1541.0 / 3780.0, atol=1e-12)
    assert np.isclose(fdr_burst[3], 1541.0 / 3780.0, atol=1e-12)


def test_full_fdr():
    samples = pl.read_parquet(FIXTURE_PATH)
    x = full_fdr(
        samples,
        csm_fdr=0.5,
        pep_fdr=0.5,
        prot_fdr=0.3,
        link_fdr=0.05,
        ppi_fdr=0.05,
        filter_back=False
    )
    assert 'csm' in x
    assert 'pep' in x
    assert 'link' in x
    assert 'ppi' in x

    # Check for _right columns and nulls in results
    for name, df in x.items():
        right_cols = [c for c in df.columns if c.endswith('_right')]
        assert not right_cols, f"Found _right columns in {name}: {right_cols}"
        if 'score' in df.columns:
            assert df['score'].null_count() == 0, f"Found null scores in {name} - check for join issues"


def test_full_fdr_consistency():
    from xifdr.fdr import pep_cols, link_cols, ppi_cols
    samples = pl.read_parquet(FIXTURE_PATH)
    results = full_fdr(
        samples,
        csm_fdr=0.1,
        pep_fdr=0.1,
        prot_fdr=0.1,
        link_fdr=0.1,
        ppi_fdr=0.1,
        filter_back=True
    )

    csm = results['csm']
    pep = results['pep']
    link = results['link']
    ppi = results['ppi']

    # 1. Check if any CSM has no corresponding peptide in the pep results
    missing_pep = csm.join(pep, on=pep_cols, how='anti')
    assert len(missing_pep) == 0, "Some CSMs do not have a corresponding Peptide match"

    # 2. Check if any peptide has no corresponding link in the link results
    # Filter out linear peptides as they don't have links
    pep_no_linear = pep.filter(pl.col('fdr_group') != 'linear')
    missing_link = pep_no_linear.join(link, on=link_cols, how='anti')
    assert len(missing_link) == 0, "Some non-linear Peptides do not have a corresponding Link match"

    # 3. Check if any link has no corresponding PPI in the ppi results
    missing_ppi = link.join(ppi, on=ppi_cols, how='anti')
    assert len(missing_ppi) == 0, "Some Links do not have a corresponding PPI match"

    # 4. Ensure no null rows were introduced by joins
    assert csm['score'].is_null().sum() == 0
    assert pep['score'].is_null().sum() == 0
    assert link['score'].is_null().sum() == 0


def test_len_filter():
    samples = pl.read_parquet(FIXTURE_PATH)
    x = full_fdr(
        samples,
        csm_fdr=1.0,
        pep_fdr=1.0,
        prot_fdr=1.0,
        link_fdr=1.0,
        ppi_fdr=1.0,
        unique_csm=False,
        min_len=8
    )
    assert len(x['csm']) < len(samples)
    assert x['csm']['sequence_p1'].str.replace_all(
        '[^A-Z]', ''
    ).str.len_chars().max() >= 8
    assert x['csm']['sequence_p2'].str.replace_all(
        '[^A-Z]', ''
    ).str.len_chars().max() >= 8
    pass

def test_full_fdr_linear():
    samples = pl.read_parquet(FIXTURE_PATH)
    samples = samples.with_columns(
        make_linear=np.random.choice([False, True], p=[0.9, 0.1], size=len(samples))
    ).with_columns(
        sequence_p2=pl.when(pl.col('make_linear')).then(
            pl.lit('')
        ).otherwise(
            pl.col('sequence_p2')
        ),
        start_pos_p2=pl.when(pl.col('make_linear')).then(
            pl.lit(None)
        ).otherwise(
            pl.col('start_pos_p2')
        ),
        link_pos_p2=pl.when(pl.col('make_linear')).then(
            pl.lit(None)
        ).otherwise(
            pl.col('link_pos_p2')
        ),
        protein_p2=pl.when(pl.col('make_linear')).then(
            pl.lit(None)
        ).otherwise(
            pl.col('protein_p2')
        ),
        decoy_p2=pl.when(pl.col('make_linear')).then(
            pl.lit(None)
        ).otherwise(
            pl.col('decoy_p2')
        ),
        coverage_p2=pl.when(pl.col('make_linear')).then(
            pl.lit(None)
        ).otherwise(
            pl.col('coverage_p2')
        ),
        fdr_group=pl.when(pl.col('make_linear')).then(
            pl.lit('linear')
        ).otherwise(
            pl.col('fdr_group')
        ),
    )
    x = full_fdr(
        samples,
        csm_fdr=0.5,
        pep_fdr=0.5,
        prot_fdr=0.3,
        link_fdr=0.05,
        ppi_fdr=0.05
    )
    pass


@pytest.mark.slow
def test_boosting():
    samples = pl.read_parquet(FIXTURE_PATH)
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
    samples = pl.read_parquet(FIXTURE_PATH)
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

def test_unpaired_fdr():
    self_tt = ([True] * 9) + [False] + ([True] * 90)
    self_td = [not x for x in self_tt]
    self_dd = [False] * 100
    between_tt = ([True] * 40)  + ([False] * 60)
    between_td = ([False] * 40) + [not not x % 3 for x in range(1, 61)]
    between_dd = ([False] * 40) + [not x % 3 for x in range(1, 61)]
    overlapping_tt = ([True] * 99) + [False]
    overlapping_td = [False] * 100
    overlapping_dd = [not x for x in overlapping_tt]

    df = pl.DataFrame(
        {
            'fdr_group': ['self'] * 100 + ['between'] * 100 + ['overlapping'] * 100,
            'TT': self_tt + between_tt + overlapping_tt,
            'TD': self_td + between_td + overlapping_td,
            'DD': self_dd + between_dd + overlapping_dd,
            'score': np.linspace(100, 0, num=300)
        }
    )

    df_grouped = df.with_columns(
        single_grouped_fdr(
            df,
            unpaired_groups=['overlapping']
        )
    )

    self_fdr = df_grouped.filter(pl.col('fdr_group') == 'self')['fdr']
    between_fdr = df_grouped.filter(pl.col('fdr_group') == 'between')['fdr']
    overlapping_fdr = df_grouped.filter(pl.col('fdr_group') == 'overlapping')['fdr']

    assert self_fdr.min() == pytest.approx(1.0 / 99.0, abs=1e-6)
    assert self_fdr.max() == pytest.approx(0.011059, abs=1e-5)
    assert between_fdr.min() == pytest.approx(3.0 / 40.0, abs=1e-6)
    assert between_fdr.max() == pytest.approx(0.258391, abs=1e-5)
    assert overlapping_fdr.min() == pytest.approx(1.0 / 99.0, abs=1e-6)
    assert overlapping_fdr.max() == pytest.approx(
        float(np.mean(1.0 / np.arange(1, 100))), abs=1e-6
    )

    # Without unpaired_groups, overlapping has no TD matches so its FDR should be 0.0
    df_default = df.with_columns(single_grouped_fdr(df))
    assert (df_default.filter(pl.col('fdr_group') == 'overlapping')['fdr'] == 0.0).all()


def _reference_single_fdr(df: pl.DataFrame, window_size: int = 5) -> np.ndarray:
    n = df.height
    if n == 0:
        return np.array([], dtype=np.float64)
    wdf = df.with_row_index('_ord').sort('score', descending=False, maintain_order=True)
    tt = [int(bool(x)) for x in wdf['TT'].fill_null(False).to_list()]
    td = [int(bool(x)) for x in wdf['TD'].fill_null(False).to_list()]
    dd = [int(bool(x)) for x in wdf['DD'].fill_null(False).to_list()]
    td_idx = [i for i, x in enumerate(td) if x]
    m = len(td_idx)
    if m == 0:
        fdr_asc = [0.0] * n
    else:
        windows = []
        if m <= window_size:
            windows.append((0, td_idx[-1]))
        else:
            windows.append((0, td_idx[window_size - 1]))
            for k in range(1, m - window_size + 1):
                windows.append((td_idx[k - 1] + 1, td_idx[k + window_size - 1]))
        vals = [[] for _ in range(n)]
        for s, e in windows:
            w_tt = sum(tt[s:e + 1])
            w_td = sum(td[s:e + 1])
            w_dd = sum(dd[s:e + 1])
            if w_tt > 0:
                v = min(1.0, max(0.0, (w_td - w_dd) / w_tt))
            else:
                v = 1.0 if (w_td - w_dd) > 0 else 0.0
            for i in range(s, e + 1):
                vals[i].append(v)
        pep = [0.0] * n
        last_td = td_idx[-1]
        for i in range(last_td + 1):
            pep[i] = sum(vals[i]) / len(vals[i])
        tail_start = windows[-1][0]
        for j in range(last_td + 1, n):
            w_tt = sum(tt[tail_start:j + 1])
            w_td = sum(td[tail_start:j + 1])
            w_dd = sum(dd[tail_start:j + 1])
            if w_tt > 0:
                pep[j] = min(1.0, max(0.0, (w_td - w_dd) / w_tt))
            else:
                pep[j] = 1.0 if (w_td - w_dd) > 0 else 0.0
        if sum(tt) == 0:
            fdr_asc = []
            cur_min = float('inf')
            for p in pep:
                cur_min = min(cur_min, p)
                fdr_asc.append(cur_min)
        else:
            global_fdr_asc = [0.0] * n
            cum_tt = 0
            cum_fp = 0.0
            for i in range(n - 1, -1, -1):
                if tt[i]:
                    cum_tt += 1
                    cum_fp += pep[i]
                global_fdr_asc[i] = (cum_fp / cum_tt) if cum_tt > 0 else float('inf')
            fdr_asc = []
            cur_min = float('inf')
            for g in global_fdr_asc:
                cur_min = min(cur_min, g)
                fdr_asc.append(cur_min)
    return wdf.with_columns(pl.Series('fdr', fdr_asc, dtype=pl.Float64)).sort('_ord')['fdr'].to_numpy()


def test_single_fdr_sliding_window_exact():
    # Shuffled input with window_size=2
    df = pl.DataFrame({
        'score': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        'TT': [True, True, False, True, True, True, False, True, False, True, True, True],
        'TD': [False, False, True, False, False, False, True, False, True, False, False, False],
        'DD': [False] * 12,
    }).sample(12, shuffle=True, seed=7)

    res = df.with_columns(single_fdr(df, window_size=2)).sort('score')
    fdr = res['fdr'].to_numpy()
    expected_from_idx11_to_0 = [
        2.0 / 7.0,
        13.0 / 42.0,
        107.0 / 315.0,
        107.0 / 315.0,
        319.0 / 840.0,
        319.0 / 840.0,
        827.0 / 2100.0,
        1016.0 / 2520.0,
        1541.0 / 3780.0,
        1541.0 / 3780.0,
        1541.0 / 3780.0,
        1541.0 / 3780.0,
    ]
    assert np.allclose(fdr, expected_from_idx11_to_0[::-1], atol=1e-12)


def test_single_fdr_matches_bruteforce():
    rng = np.random.default_rng(123)
    for n in [1, 2, 5, 12, 30, 50]:
        for window_size in [1, 2, 5, 10]:
            cls = rng.choice([0, 1, 2], size=n, p=[0.45, 0.35, 0.20])
            df = pl.DataFrame({
                'score': rng.uniform(0, 100, size=n),
                'TT': (cls == 0).tolist(),
                'TD': (cls == 1).tolist(),
                'DD': (cls == 2).tolist(),
            })
            actual = single_fdr(df, window_size=window_size).to_numpy()
            expected = _reference_single_fdr(df, window_size=window_size)
            assert np.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_single_fdr_edge_cases():
    # Empty DataFrame
    empty_df = pl.DataFrame(
        schema={'score': pl.Float64, 'TT': pl.Boolean, 'TD': pl.Boolean, 'DD': pl.Boolean}
    )
    empty_res = single_fdr(empty_df)
    assert empty_res.len() == 0
    assert empty_res.dtype == pl.Float64

    # Zero TDs -> all 0.0
    no_td_df = pl.DataFrame({
        'score': [1.0, 2.0, 3.0],
        'TT': [True, False, True],
        'TD': [False, False, False],
        'DD': [False, True, False],
    })
    assert (single_fdr(no_td_df) == 0.0).all()

    # Fewer TDs than window_size (single full window)
    few_td_df = pl.DataFrame({
        'score': [1.0, 2.0, 3.0, 4.0],
        'TT': [True, False, True, True],
        'TD': [False, True, False, False],
        'DD': [False, False, False, False],
    })
    # Window [0..1]: TD=1, TT=1 -> pep=1.0; tail idx 2: [0..2] TD=1, TT=2 -> 0.5; idx 3: [0..3] TD=1, TT=3 -> 1/3
    # Cumulative from top: idx 3: 1/3; idx 2: (1/3 + 1/2)/2 = 5/12; idx 1: 5/12; idx 0: (1/3 + 1/2 + 1)/3 = 11/18
    assert np.allclose(
        single_fdr(few_td_df, window_size=5).to_numpy(),
        [11.0 / 18.0, 5.0 / 12.0, 5.0 / 12.0, 1.0 / 3.0],
        atol=1e-12,
    )

    # Zero TTs with TD > DD and TD <= DD
    no_tt_pos = pl.DataFrame({
        'score': [1.0, 2.0],
        'TT': [False, False],
        'TD': [True, False],
        'DD': [False, True],
    })
    assert np.allclose(single_fdr(no_tt_pos, window_size=1).to_numpy(), [1.0, 0.0], atol=1e-12)

    no_tt_neg = pl.DataFrame({
        'score': [1.0, 2.0],
        'TT': [False, False],
        'TD': [False, True],
        'DD': [True, False],
    })
    assert np.allclose(single_fdr(no_tt_neg, window_size=1).to_numpy(), [0.0, 0.0], atol=1e-12)

    # Highest-scoring row is TD (no tail after last TD)
    top_td_df = pl.DataFrame({
        'score': [1.0, 2.0, 3.0],
        'TT': [True, True, False],
        'TD': [False, False, True],
        'DD': [False, False, False],
    })
    # Window [0..2]: TD=1, TT=2 -> 0.5 everywhere
    assert np.allclose(single_fdr(top_td_df, window_size=5).to_numpy(), [0.5, 0.5, 0.5], atol=1e-12)

    # Clipping negative (DD > TD) and TT == 0 windows before averaging
    clip_df = pl.DataFrame({
        'score': [1.0, 2.0, 3.0, 4.0],
        'TT': [False, False, False, True],
        'TD': [True, True, True, False],
        'DD': [False, False, False, False],
    })
    # W0=[0..0] -> 1.0; W1=[1..1] -> 1.0; W2=[2..2] -> 1.0; tail idx 3 [2..3] -> 1/1 = 1.0
    assert np.allclose(single_fdr(clip_df, window_size=1).to_numpy(), [1.0, 1.0, 1.0, 1.0], atol=1e-12)

    dd_gt_td_df = pl.DataFrame({
        'score': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'TT': [True, False, False, False, True, True],
        'TD': [False, True, False, False, True, False],
        'DD': [False, False, True, True, False, False],
    })
    # W0=[0..1]: TD=1,DD=0,TT=1 -> 1.0
    # W1=[2..4]: TD=1,DD=2,TT=1 -> -1.0 -> clipped to 0.0 before averaging!
    # Tail idx 5 [2..5]: TD=1,DD=2,TT=2 -> -0.5 -> clipped to 0.0
    # So pep = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0], global at idx 5,4=0.0, idx 0=1/3
    assert np.allclose(
        single_fdr(dd_gt_td_df, window_size=1).to_numpy(),
        [1.0 / 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        atol=1e-12,
    )

    # Invalid window_size
    with pytest.raises(ValueError):
        single_fdr(top_td_df, window_size=0)

    # Pandas DataFrame input and pre-existing 'order_col'
    pd_df = few_td_df.with_columns(pl.lit(99).alias('order_col')).to_pandas()
    pd_res = single_fdr(pd_df, window_size=5)
    assert isinstance(pd_res, pl.Series)
    assert pd_res.name == 'fdr'
    assert np.allclose(pd_res.to_numpy(), [11.0 / 18.0, 5.0 / 12.0, 5.0 / 12.0, 1.0 / 3.0], atol=1e-12)

    # window_size propagation through single_grouped_fdr
    grouped_df = pl.concat([
        dd_gt_td_df.with_columns(pl.lit('g1').alias('fdr_group')),
        top_td_df.with_columns(pl.lit('g2').alias('fdr_group')),
    ])
    g_w1 = single_grouped_fdr(grouped_df, window_size=1)
    g_w5 = single_grouped_fdr(grouped_df, window_size=5)
    assert g_w1.dtype == pl.Float64
    assert np.allclose(
        g_w1.to_numpy(),
        np.concatenate([
            single_fdr(dd_gt_td_df, window_size=1).to_numpy(),
            single_fdr(top_td_df, window_size=1).to_numpy(),
        ]),
        atol=1e-12,
    )
    assert np.allclose(
        g_w5.to_numpy(),
        np.concatenate([
            single_fdr(dd_gt_td_df, window_size=5).to_numpy(),
            single_fdr(top_td_df, window_size=5).to_numpy(),
        ]),
        atol=1e-12,
    )
    assert not np.allclose(g_w1.to_numpy(), g_w5.to_numpy())
