import string
import warnings

import polars as pl

from xifdr.fdr import full_fdr, _csm_fdr, _pep_fdr, _prot_fdr, _link_fdr, _ppi_fdr


def test_standard_td_prob():
    agg = (pl.col('score')**2).sum().sqrt()
    never_aggs = ['fdr_group', 'decoy_class', 'TT', 'TD', 'DD', 'score', 'protein_score_p1', 'protein_score_p2']
    first_aggs = [
        pl.col('TT').get(0),
        pl.col('TD').get(0),
        pl.col('DD').get(0),
        pl.col('fdr_group').get(0),
    ]
    samples = pl.read_parquet('tests/fixtures/sample_data.parquet')
    x = full_fdr(
        samples,
        csm_fdr=0.5,
        pep_fdr=0.5,
        prot_fdr=0.5,
        link_fdr=0.5,
        ppi_fdr=0.5
    )
    for fdr_group in ['self', 'between', 'linear']:
        # Test too few TT in CSM level
        n_tt = min(len(x['csm'].filter('TT', pl.col('fdr_group') == fdr_group)), 99)
        n_td = min(len(x['csm'].filter('TD', pl.col('fdr_group') == fdr_group)), 20)
        n_dd = min(len(x['csm'].filter('DD', pl.col('fdr_group') == fdr_group)), 10)
        df = pl.concat([
            x['csm'].filter('TT', pl.col('fdr_group') == fdr_group).sample(n_tt, seed=0),
            x['csm'].filter('TD', pl.col('fdr_group') == fdr_group).sample(n_td, seed=0),
            x['csm'].filter('DD', pl.col('fdr_group') == fdr_group).sample(n_dd, seed=0),
            x['csm'].filter(pl.col('fdr_group') != fdr_group)
        ])
        res = _csm_fdr(df, 0.01, True, td_prob=1, td_dd_ratio=0)
        assert res.filter(pl.col('fdr_group') == fdr_group).height == 0

        # Test too few TT in peptide level
        n_tt = min(len(x['csm'].filter('TT', pl.col('fdr_group') == fdr_group)), 99)
        n_td = min(len(x['csm'].filter('TD', pl.col('fdr_group') == fdr_group)), 20)
        n_dd = min(len(x['csm'].filter('DD', pl.col('fdr_group') == fdr_group)), 10)
        df = pl.concat([
            x['csm'].filter('TT', pl.col('fdr_group') == fdr_group).sample(n_tt, seed=0),
            x['csm'].filter('TD', pl.col('fdr_group') == fdr_group).sample(n_td, seed=0),
            x['csm'].filter('DD', pl.col('fdr_group') == fdr_group).sample(n_dd, seed=0),
            x['csm'].filter(pl.col('fdr_group') != fdr_group)
        ])
        res = _pep_fdr(
            df, agg, 0.01,
            first_aggs=first_aggs, never_agg_cols=never_aggs,
            td_prob=1, td_dd_ratio=0
        )
        assert res.filter(pl.col('fdr_group') == fdr_group).height == 0

    for fdr_group in ['self', 'between']:
        # Test too few TT in link level
        n_tt = min(len(x['pep'].filter('TT', pl.col('fdr_group') == fdr_group)), 99)
        n_td = min(len(x['pep'].filter('TD', pl.col('fdr_group') == fdr_group)), 20)
        n_dd = min(len(x['pep'].filter('DD', pl.col('fdr_group') == fdr_group)), 10)
        df = pl.concat([
            x['pep'].filter('TT', pl.col('fdr_group') == fdr_group).sample(n_tt, seed=0),
            x['pep'].filter('TD', pl.col('fdr_group') == fdr_group).sample(n_td, seed=0),
            x['pep'].filter('DD', pl.col('fdr_group') == fdr_group).sample(n_dd, seed=0),
            x['pep'].filter(pl.col('fdr_group') != fdr_group)
        ])
        res = _link_fdr(
            df, agg, 0.01,
            first_aggs=first_aggs, never_agg_cols=never_aggs,
            td_prob=1, td_dd_ratio=0
        )
        assert res.filter(pl.col('fdr_group') == fdr_group).height == 0

        # Test too few TT in PPI level
        n_tt = min(len(x['link'].filter('TT', pl.col('fdr_group') == fdr_group)), 99)
        n_td = min(len(x['link'].filter('TD', pl.col('fdr_group') == fdr_group)), 20)
        n_dd = min(len(x['link'].filter('DD', pl.col('fdr_group') == fdr_group)), 10)
        df = pl.concat([
            x['link'].filter('TT', pl.col('fdr_group') == fdr_group).sample(n_tt, seed=0),
            x['link'].filter('TD', pl.col('fdr_group') == fdr_group).sample(n_td, seed=0),
            x['link'].filter('DD', pl.col('fdr_group') == fdr_group).sample(n_dd, seed=0),
            x['link'].filter(pl.col('fdr_group') != fdr_group)
        ])
        res = _ppi_fdr(
            df, agg, 0.01,
            first_aggs=first_aggs, never_agg_cols=never_aggs,
            td_prob=1, td_dd_ratio=0
        )
        assert res.filter(pl.col('fdr_group') == fdr_group).height == 0

def test_protein_td_inval_merge():
    """
    Test that the protein level FDR keeps unsupported and linear/self matches
    due to sufficient TT matches when joined.
    """
    agg = (pl.col('score')**2).sum().sqrt()
    # 6 unsupported
    df_unsup = pl.DataFrame({
        'protein_p1': list(string.ascii_uppercase)[0:5],
        'protein_p2': list(string.ascii_uppercase)[1:6],
        'fdr_group': ['between']*(5-0),
    })
    # 6 self
    df_self = pl.DataFrame({
        'protein_p1': list(string.ascii_uppercase)[7:13],
        'protein_p2': list(string.ascii_uppercase)[7:13],
        'fdr_group': ['self']*(13-7),
    })
    # 10 supported + 1 self
    df_sup_self = pl.DataFrame({
        'protein_p1': list(string.ascii_uppercase)[15:25],
        'protein_p2': list(string.ascii_uppercase)[15:25],
        'fdr_group': ['self']*(25-15),
    })
    df_sup_between = pl.DataFrame({
        'protein_p1': list(string.ascii_uppercase)[15:25],
        'protein_p2': list(string.ascii_uppercase)[16:26],
        'fdr_group': ['between']*(25-15),
    })
    df = pl.concat([
        df_unsup,
        df_self,
        df_sup_self,
        df_sup_between,
    ])
    df = df.with_columns(
        pl.col('protein_p1').str.split(''),
        pl.col('protein_p2').str.split(''),
        decoy_p1=pl.lit(False),
        decoy_p2=pl.lit(False),
        protein_score_p1=pl.lit(0),
        protein_score_p2=pl.lit(0),
    )
    res = _prot_fdr(df, agg, 0.1, 1)
    assert res.filter(pl.col('protein_fdr_group') == 'supported_between').height > 0
    assert res.filter(pl.col('protein_fdr_group') == 'unsupported_between').height == 0
    assert res.filter(pl.col('protein_fdr_group') == 'self_or_linear').height == 0
    assert res.filter(pl.col('protein_fdr_group') == 'invalid_merged').height > 0

def test_protein_td_only_supp():
    """
    Test that the protein level FDR keeps only unsupported and not linear/self matches
    due to insufficient TT matches.
    """
    agg = (pl.col('score')**2).sum().sqrt()
    # 4 unsupported
    df_unsup = pl.DataFrame({
        'protein_p1': list(string.ascii_uppercase)[0:3],
        'protein_p2': list(string.ascii_uppercase)[1:4],
        'fdr_group': ['between']*(3-0),
    })
    # 4 self
    df_self = pl.DataFrame({
        'protein_p1': list(string.ascii_uppercase)[7:11],
        'protein_p2': list(string.ascii_uppercase)[7:11],
        'fdr_group': ['self']*(11-7),
    })
    # 10 supported + 1 self
    df_sup_self = pl.DataFrame({
        'protein_p1': list(string.ascii_uppercase)[15:25],
        'protein_p2': list(string.ascii_uppercase)[15:25],
        'fdr_group': ['self']*(25-15),
    })
    df_sup_between = pl.DataFrame({
        'protein_p1': list(string.ascii_uppercase)[15:25],
        'protein_p2': list(string.ascii_uppercase)[16:26],
        'fdr_group': ['between']*(25-15),
    })
    df = pl.concat([
        df_unsup,
        df_self,
        df_sup_self,
        df_sup_between,
    ])
    df = df.with_columns(
        pl.col('protein_p1').str.split(''),
        pl.col('protein_p2').str.split(''),
        decoy_p1=pl.lit(False),
        decoy_p2=pl.lit(False),
        protein_score_p1=pl.lit(0),
        protein_score_p2=pl.lit(0),
    )
    res = _prot_fdr(df, agg, 0.1, 1)
    assert res.filter(pl.col('protein_fdr_group') == 'supported_between').height > 0
    assert res.filter(pl.col('protein_fdr_group') == 'unsupported_between').height == 0
    assert res.filter(pl.col('protein_fdr_group') == 'self_or_linear').height == 0
    assert res.filter(pl.col('protein_fdr_group') == 'invalid_merged').height == 0

def test_td_dd_retio():
    warnings.simplefilter("always")
    agg = (pl.col('score')**2).sum().sqrt()
    never_aggs = ['fdr_group', 'decoy_class', 'TT', 'TD', 'DD', 'score', 'protein_score_p1', 'protein_score_p2']
    first_aggs = [
        pl.col('TT').get(0),
        pl.col('TD').get(0),
        pl.col('DD').get(0),
        pl.col('fdr_group').get(0),
    ]
    samples = pl.read_parquet('tests/fixtures/sample_data.parquet')
    x = full_fdr(
        samples,
        csm_fdr=0.5,
        pep_fdr=0.5,
        prot_fdr=0.5,
        link_fdr=0.5,
        ppi_fdr=0.5
    )
    for fdr_group in ['self', 'between']:
        # Test too few TT in CSM level
        n_td = min(len(x['csm'].filter('TD', pl.col('fdr_group') == fdr_group)), 10)
        n_dd = min(len(x['csm'].filter('DD', pl.col('fdr_group') == fdr_group)), 20)
        df = pl.concat([
            x['csm'].filter('TT', pl.col('fdr_group') == fdr_group),
            x['csm'].filter('TD', pl.col('fdr_group') == fdr_group).sample(n_td, seed=0),
            x['csm'].filter('DD', pl.col('fdr_group') == fdr_group).sample(n_dd, seed=0),
            x['csm'].filter(pl.col('fdr_group') != fdr_group)
        ])
        with warnings.catch_warnings(record=True) as caught_warns:
            res = _csm_fdr(
                df, 0.01, True,
                td_prob=1, td_dd_ratio=1
            )
            assert res.filter(pl.col('fdr_group') == fdr_group).height == 0
            td_dd_warn = [
                w.message for w in caught_warns
                if str(w.message) == f'More DD than TD for CSM FDR in group {fdr_group}.'
            ]
            assert len(td_dd_warn) == 1

        # Test too few TT in peptide level
        n_td = min(len(x['csm'].filter('TD', pl.col('fdr_group') == fdr_group)), 10)
        n_dd = min(len(x['csm'].filter('DD', pl.col('fdr_group') == fdr_group)), 20)
        df = pl.concat([
            x['csm'].filter('TT', pl.col('fdr_group') == fdr_group),
            x['csm'].filter('TD', pl.col('fdr_group') == fdr_group).sample(n_td, seed=0),
            x['csm'].filter('DD', pl.col('fdr_group') == fdr_group).sample(n_dd, seed=0),
            x['csm'].filter(pl.col('fdr_group') != fdr_group)
        ])
        with warnings.catch_warnings(record=True) as caught_warns:
            res = _pep_fdr(
                df, agg, 0.01,
                first_aggs=first_aggs, never_agg_cols=never_aggs,
                td_dd_ratio=1, td_prob=1
            )
            assert res.filter(pl.col('fdr_group') == fdr_group).height == 0
            td_dd_warn = [
                w.message for w in caught_warns
                if str(w.message) == f'More DD than TD for peptide FDR in group {fdr_group}.'
            ]
            assert len(td_dd_warn) == 1

    for fdr_group in ['self', 'between']:
        # Test too few TT in link level
        n_td = min(len(x['pep'].filter('TD', pl.col('fdr_group') == fdr_group)), 10)
        n_dd = min(len(x['pep'].filter('DD', pl.col('fdr_group') == fdr_group)), 20)
        df = pl.concat([
            x['pep'].filter('TT', pl.col('fdr_group') == fdr_group),
            x['pep'].filter('TD', pl.col('fdr_group') == fdr_group).sample(n_td, seed=0),
            x['pep'].filter('DD', pl.col('fdr_group') == fdr_group).sample(n_dd, seed=0),
            x['pep'].filter(pl.col('fdr_group') != fdr_group)
        ])
        with warnings.catch_warnings(record=True) as caught_warns:
            res = _link_fdr(
                df, agg, 0.01,
                first_aggs=first_aggs, never_agg_cols=never_aggs,
                td_prob=1, td_dd_ratio=1
            )
            assert res.filter(pl.col('fdr_group') == fdr_group).height == 0
            td_dd_warn = [
                w.message for w in caught_warns
                if str(w.message) == f'More DD than TD for link FDR in group {fdr_group}.'
            ]
            assert len(td_dd_warn) == 1

        # Test too few TT in PPI level
        n_td = min(len(x['link'].filter('TD', pl.col('fdr_group') == fdr_group)), 10)
        n_dd = min(len(x['link'].filter('DD', pl.col('fdr_group') == fdr_group)), 20)
        df = pl.concat([
            x['link'].filter('TT', pl.col('fdr_group') == fdr_group),
            x['link'].filter('TD', pl.col('fdr_group') == fdr_group).sample(n_td, seed=0),
            x['link'].filter('DD', pl.col('fdr_group') == fdr_group).sample(n_dd, seed=0),
            x['link'].filter(pl.col('fdr_group') != fdr_group)
        ])
        with warnings.catch_warnings(record=True) as caught_warns:
            res = _ppi_fdr(
                df, agg, 0.01,
                first_aggs=first_aggs, never_agg_cols=never_aggs,
                td_prob=1, td_dd_ratio=1
            )
            assert res.filter(pl.col('fdr_group') == fdr_group).height == 0
            td_dd_warn = [
                w.message for w in caught_warns
                if str(w.message) == f'More DD than TD for PPI FDR in group {fdr_group}.'
            ]
            assert len(td_dd_warn) == 1
