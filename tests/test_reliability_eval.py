import pytest
import polars as pl

from xifdr.fdr import _csm_fdr, _pep_fdr, _link_fdr, _ppi_fdr


def test_td_prob():
    df = pl.DataFrame(
        {
            'fdr_group': (['A'] * 40) + (['B'] * 10) + (['C'] * 50),
            'score': [1.0] * 100,
            'TT': ([1] * 50) + ([1] * 40) + ([0] * 10),
            'TD': ([0] * 50) + ([0] * 40) + ([1] * 3) + ([0] * 7),
            'DD': ([0] * 50) + ([0] * 40) + ([0] * 3) + ([1] * 7),
        }
    )

    csm_df = _csm_fdr(
        df, 0.1, unique_csm=False, td_prob=2, td_dd_ratio=0.5
    )
    csm_df_a = csm_df.filter(pl.col('fdr_group') == 'A')
    csm_df_b = csm_df.filter(pl.col('fdr_group') == 'B')
    csm_df_c = csm_df.filter(pl.col('fdr_group') == 'C')
    assert csm_df_a['csm_td_check'].all()
    assert csm_df_a['csm_dd_check'].all()
    assert not csm_df_b['csm_td_check'].any()
    assert csm_df_b['csm_dd_check'].all()
    assert csm_df_c['csm_td_check'].all()
    assert not csm_df_c['csm_dd_check'].any()
