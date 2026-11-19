import logging
import warnings
from typing import Union

import numpy as np
import pandas as pd
import polars as pl
from xifdr.utils.column_preparation import prepare_columns
from xifdr.utils import expression_utils


logger = logging.getLogger(__name__)

csm_cols = [
    'decoy_p1', 'decoy_p2', 'sequence_p1', 'sequence_p2',
    'protein_p1', 'protein_p2', 'cl_pos_p1', 'cl_pos_p2',
    'ppi_protein_p1', 'ppi_protein_p2', 'charge'
]
pep_cols = [
    'decoy_p1', 'decoy_p2', 'sequence_p1', 'sequence_p2', 'protein_p1', 'protein_p2',
    'cl_pos_p1', 'cl_pos_p2', 'ppi_protein_p1', 'ppi_protein_p2',
]
link_cols = [
    'decoy_p1', 'decoy_p2', 'protein_p1', 'protein_p2', 'cl_pos_p1', 'cl_pos_p2',
    'ppi_protein_p1', 'ppi_protein_p2',
]
ppi_cols = ['decoy_p1', 'decoy_p2', 'ppi_protein_p1', 'ppi_protein_p2',]
fdr_groups_csm_pep = ['self', 'between', 'linear']  # FDR groups for CSM and peptide level
fdr_groups_link_ppi = ['self', 'between']  # FDR groups for link and PPI level


def full_fdr(df: Union[pl.DataFrame, pd.DataFrame],
             csm_fdr:float = 1.0,
             pep_fdr:float = 1.0,
             prot_fdr:float = 1.0,
             link_fdr:float = 1.0,
             ppi_fdr:float = 1.0,
             min_len:int = 5,
             decoy_adjunct:str = 'REV_',
             unique_csm:bool = True,
             filter_back:bool = True,
             prepare_column:bool = True,
             td_prob:int = 2,
             td_prot_prob:int = 10,
             td_dd_ratio:float = 1.0,
             custom_aggs:dict = None,
             window_size: int = 5) -> dict[str, pl.DataFrame]:
    """
    
    Parameters
    ----------
    df
        Input CSM dataframe
    csm_fdr
        CSM level FDR cutoff
    pep_fdr
        Peptide level FDR cutoff
    prot_fdr
        Protein level FDR cutoff
    link_fdr
        Link level FDR cutoff
    ppi_fdr
        Protein pair level FDR cutoff
    min_len
        Minimum peptide sequence length
    decoy_adjunct
        Prefix/Suffix indicating a decoy match
    unique_csm
        Make CSMs unique
    filter_back
        Filter lower levels to include only matches that also pass on higher levels
    prepare_column
        Perform preparation of aggregation columns like sorting ambiguous proteins and swapping protein 1/2
    td_prob
        Minimum theoretical TD machtes for the FDR levels (except protein level)
    td_prot_prob
        Minimum theoretical TD machtes for the protein FDR level
    td_dd_ratio
        Minimum ratio of TD/DD
    custom_aggs
        Custom aggregation functions for the FDR levels
    window_size
        Number of TD matches per sliding window for interpolated decoy FDR estimation

    Returns
    -------
        Return a dict with keys `'csm'`, `'pep'`, `'prot'`, `'link'`, `'ppi'` that contains the resulting polars DataFrame for each FDR level.
    """
    aggs = {
        'pep': (pl.col('score')**2).sum().sqrt(),
        'prot': (pl.col('score')**2).sum().sqrt(),
        'link': (pl.col('score')**2).sum().sqrt(),
        'ppi': (pl.col('score')**2).sum().sqrt(),
    }
    if custom_aggs is not None:
        aggs.update(custom_aggs)

    if prepare_column:
        df = prepare_columns(df)

    # Filter CSMs for minimum peptide length
    df = df.filter(
        pl.col('sequence_p1').str.replace_all('[^A-Z]', '').str.len_chars() >= min_len,
        pl.col('sequence_p2').str.replace_all('[^A-Z]', '').str.len_chars() >= min_len,
    )

    # Check for required columns
    required_columns = [
        'score',  # Match score
        'decoy_p1', 'decoy_p2',  # Target/decoy classification
        'charge',  # Precursor charge
        'start_pos_p1', 'start_pos_p2',  # Position of peptides in proteins origins
        'link_pos_p1', 'link_pos_p2',  # Position of the link in the peptides
        'sequence_p1', 'sequence_p2',  # Peptide sequences including modifications
        'protein_p1', 'protein_p2',  # Protein origins of the peptides
    ]

    # Check for required columns
    missing_columns = [
        c for c in required_columns
        if c not in df.columns
    ]
    if len(missing_columns) > 0:
        raise Exception(f'Missing required columns: {missing_columns}')

    # Aggregate unique CSMs
    never_agg_cols = ['fdr_group', 'decoy_class', 'TT', 'TD', 'DD']
    first_aggs = [
        pl.col(c).get(0)
        for c in never_agg_cols
    ]
    never_agg_cols += ['score', 'protein_score_p1', 'protein_score_p2']

    df_csm = _csm_fdr(df, csm_fdr, unique_csm, td_prob, td_dd_ratio, window_size=window_size)

    # Calculate peptide FDR and filter
    logger.debug('Calculate peptide FDR and filter')
    df_pep = _pep_fdr(df_csm, aggs['pep'], pep_fdr, first_aggs, never_agg_cols, td_prob, td_dd_ratio, window_size=window_size)

    logger.debug('Calculate protein FDR and filter')
    df_prot = _prot_fdr(df_pep, aggs['prot'], prot_fdr, td_prot_prob, window_size=window_size)

    logger.debug('Filter peptide pairs for passed proteins')
    df_pep = _prot_filter(df_pep, df_prot, decoy_adjunct)

    # Calculate link FDR and cutoff
    logger.debug('Calculate link FDR and cutoff')
    df_link = _link_fdr(df_pep, aggs['link'], link_fdr, first_aggs, never_agg_cols, td_prob, td_dd_ratio, window_size=window_size)

    # Calculate PPI FDR
    logger.debug('Calculate PPI FDR')
    df_ppi = _ppi_fdr(df_link, aggs['prot'], ppi_fdr, first_aggs, never_agg_cols, td_prob, td_dd_ratio, window_size=window_size)

    # Back-fitler levels
    df_ppi = df_ppi.with_columns(pass_threshold=pl.lit(True))

    pass_on_cols = [pl.col('score').alias('ppi_score'), 'ppi_fdr', 'pass_threshold']
    df_link = df_link.join(
        df_ppi.select(*ppi_cols, *pass_on_cols),
        on=ppi_cols,
        how='full',
        coalesce=True,
    ).with_columns(
        pass_threshold=pl.col('pass_threshold').fill_null(pl.lit(False))
    )

    pass_on_cols += [pl.col('score').alias('link_score'), 'link_fdr']
    df_pep = df_pep.join(
        df_link.select(*link_cols, *pass_on_cols),
        on=link_cols,
        how='full',
        coalesce=True,
    ).with_columns(
        pass_threshold=pl.col('pass_threshold').fill_null(pl.lit(False))
    )

    pass_on_cols += [pl.col('score').alias('pep_score'), 'pep_fdr']
    df_csm = df_csm.join(
        df_pep.select(*pep_cols, *pass_on_cols),
        on=pep_cols,
        how='full',
        coalesce=True,
    ).with_columns(
        pass_threshold=pl.col('pass_threshold').fill_null(pl.lit(False))
    )

    if filter_back:
        df_link = df_link.filter('pass_threshold')
        df_pep = df_pep.filter('pass_threshold')
        df_csm = df_csm.filter('pass_threshold')

    df_ppi = df_ppi.with_columns(
        protein_p1='ppi_protein_p1',
        protein_p2='ppi_protein_p2'
    )

    return {
        'csm': df_csm,
        'pep': df_pep,
        'prot': df_prot,
        'link': df_link,
        'ppi': df_ppi,
    }

def _csm_fdr(df, csm_fdr, unique_csm, td_prob, td_dd_ratio, window_size: int = 5):
    if unique_csm:
        df_csm = df.sort('score', descending=True).unique(subset=csm_cols, keep='first')
    else:
        df_csm = df

    # Calculate CSM FDR and cutoff
    logger.debug('Calculate CSM FDR and cutoff')
    df_csm = df_csm.with_columns(
        csm_fdr = single_grouped_fdr(df_csm, window_size=window_size)
    ).filter(pl.col('csm_fdr').clip(0.0, 1.0) <= csm_fdr)

    df_csm_checks = df_csm.group_by('fdr_group').agg(
        pl.col('TD').sum().alias('n_td'),
        pl.col('DD').sum().alias('n_dd'),
        pl.col('TT').sum().alias('n_tt'),
        pl.all(),
    ).with_columns(
        csm_td_check = pl.col('n_tt') * csm_fdr >= td_prob,
        csm_dd_check = pl.col('n_dd') * td_dd_ratio <= pl.col('n_td'),
    )

    for row in df_csm_checks.to_dicts():
        fdr_group = row['fdr_group']
        td_check = row['csm_td_check']
        dd_check = row['csm_dd_check']
        if not td_check:
            warnings.warn(f'Insufficient TT for CSM FDR in group {fdr_group}.')
        if not dd_check:
            warnings.warn(f'More DD than TT for CSM FDR in group {fdr_group}.')

    df_csm = df_csm_checks.explode(
        pl.selectors.list()
    ).select(
        *df_csm.columns,
        'csm_td_check',
        'csm_dd_check',
    )

    return df_csm


def _pep_fdr(df_csm, agg, pep_fdr, first_aggs, never_agg_cols, td_prob, td_dd_ratio, window_size: int = 5):
    pep_merge_cols = [c for c in df_csm.columns if c not in pep_cols+never_agg_cols]
    df_pep = df_csm.group_by(pep_cols).agg(
        *first_aggs,
        *[
            pl.col(c).list.explode()
            for c in pep_merge_cols
        ],
        protein_score_p1=expression_utils.replace_input(agg, 'protein_score_p1'),
        protein_score_p2=expression_utils.replace_input(agg, 'protein_score_p2'),
        score=agg
    )
    df_pep = df_pep.with_columns(
        pep_fdr = single_grouped_fdr(df_pep, window_size=window_size)
    ).filter(pl.col('pep_fdr').clip(0.0, 1.0) <= pep_fdr)

    df_pep_checks = df_pep.group_by('fdr_group').agg(
        pl.col('TD').sum().alias('n_td'),
        pl.col('DD').sum().alias('n_dd'),
        pl.col('TT').sum().alias('n_tt'),
        pl.all(),
    ).with_columns(
        pep_td_check = pl.col('n_tt') * pep_fdr >= td_prob,
        pep_dd_check = pl.col('n_dd') * td_dd_ratio <= pl.col('n_td'),
    )

    for row in df_pep_checks.to_dicts():
        fdr_group = row['fdr_group']
        td_check = row['pep_td_check']
        dd_check = row['pep_dd_check']
        if not td_check:
            warnings.warn(f'Insufficient TT for peptide FDR in group {fdr_group}.')
        if not dd_check:
            warnings.warn(f'More DD than TT for peptide FDR in group {fdr_group}.')

    df_pep = df_pep_checks.explode(
        pl.selectors.list()
    ).select(
        *df_pep.columns,
        'pep_td_check',
        'pep_dd_check',
    )

    return df_pep


def _prot_fdr(df_pep:pl.DataFrame,
              agg,
              prot_fdr,
              td_prot_prob,
              window_size: int = 5) -> pl.DataFrame:
    # Construct protein (group) DF
    df_prot_p1 = df_pep.select([
        'protein_p1', 'protein_score_p1', 'decoy_p1', 'fdr_group'
    ]).rename({
        'protein_p1': 'protein',
        'protein_score_p1': 'score',
        'decoy_p1': 'decoy',
    })

    df_prot_p2 = df_pep.select([
        'protein_p2', 'protein_score_p2', 'decoy_p2', 'fdr_group'
    ]).rename({
        'protein_p2': 'protein',
        'protein_score_p2': 'score',
        'decoy_p2': 'decoy',
    })

    df_prot = pl.concat([
        df_prot_p1,
        df_prot_p2
    ])
    df_prot = df_prot.with_columns(
        protein_group=pl.col('protein').list.unique().list.sort()
    )
    df_prot = df_prot.group_by(['protein_group', 'decoy']).agg(
        pl.col('protein'),
        pl.col('fdr_group'),
        score=agg
    ).with_columns(
        no_self=~pl.lit('self').is_in(pl.col('fdr_group')),
        no_overlapping=~pl.lit('overlapping').is_in(pl.col('fdr_group')),
        no_linear=~pl.lit('linear').is_in(pl.col('fdr_group')),
        between=pl.lit('between').is_in(pl.col('fdr_group')),
    ).with_columns(
        protein_fdr_group=(
            pl.when(pl.col('between') & pl.col('no_self') & pl.col('no_overlapping') & pl.col('no_linear'))
            .then(pl.lit('unsupported_between'))
            .otherwise(pl.lit('self_linear_supported'))
        )
    )
    df_prot = df_prot.with_columns(
        TD=pl.col('decoy'),
        TT=~pl.col('decoy'),
        DD=pl.lit(False)  # Abuse CL-FDR for linear case
    )
    df_prot = df_prot.with_columns(
        prot_fdr=single_grouped_fdr(df_prot, fdr_group_col='protein_fdr_group', window_size=window_size)
    )
    df_prot = df_prot.filter(pl.col('prot_fdr').clip(0.0, 1.0) <= prot_fdr)

    # Check whether there are at least enough TT to have approx. `min_td` TD matches under the requested FDR level.
    df_prot_checks = df_prot.group_by('protein_fdr_group').agg(
        pl.col('TT').sum().alias('n_t'),
        pl.all(),
    ).with_columns(
        prot_td_check = pl.col('n_t') * prot_fdr >= td_prot_prob,
    )

    df_prot = df_prot_checks.explode(
        pl.selectors.list()
    ).select(
        *df_prot.columns,
        'prot_td_check',
    )

    return df_prot


def _prot_filter(df_pep, df_prot, decoy_adjunct):
    passed_prots = df_prot['protein'].explode()
    passed_prots = passed_prots.list.join(';')
    passed_prots = passed_prots.str.replace_all(decoy_adjunct, '')
    passed_prots = passed_prots.str.split(';')
    passed_prots = passed_prots.list.sort()
    passed_prots = passed_prots.list.join(';')
    passed_prots = passed_prots.unique().alias('passed_prots')

    ## Filter left over peptide pairs
    logger.debug('Filter left over peptide pairs')
    df_pep = df_pep.with_columns(
        base_protein_p1 = (
            pl.col('protein_p1')
                # Replace decoy_adjunct
                .list.join(';')
                .str.replace_all(decoy_adjunct, '')
                # Sort base protein names
                .str.split(';')
                .list.sort()
                # Join to protein group
                .list.join(';')
        ),
        base_protein_p2 = (
            pl.col('protein_p2')
                # Replace decoy_adjunct
                .list.join(';')
                .str.replace_all(decoy_adjunct, '')
                # Sort base protein names
                .str.split(';')
                .list.sort()
                # Join to protein group
                .list.join(';')
        ),
    )

    return df_pep.join(
        passed_prots.to_frame(),
        left_on=['base_protein_p1'],
        right_on=['passed_prots'],
        how='inner',
        suffix='p1'
    ).join(
        passed_prots.to_frame(),
        left_on=['base_protein_p2'],
        right_on=['passed_prots'],
        how='inner',
        suffix='p2'
    )


def _link_fdr(df_pep, agg, link_fdr, first_aggs, never_agg_cols, td_prob, td_dd_ratio, window_size: int = 5):
    link_merge_cols = [c for c in df_pep.columns if c not in link_cols+never_agg_cols]
    df_link = df_pep.filter(
        pl.col('fdr_group') != "linear" # Disregard linear peptides from here on
    ).group_by(link_cols).agg(
        *first_aggs,
        *[
            pl.col(c).list.explode()
            for c in link_merge_cols
        ],
        score=agg
    )
    df_link = df_link.with_columns(
        link_fdr = single_grouped_fdr(df_link, window_size=window_size)
    ).filter(pl.col('link_fdr').clip(0.0, 1.0) <= link_fdr)

    df_link_checks = df_link.group_by('fdr_group').agg(
        pl.col('TD').sum().alias('n_td'),
        pl.col('DD').sum().alias('n_dd'),
        pl.col('TT').sum().alias('n_tt'),
        pl.all(),
    ).with_columns(
        link_td_check = pl.col('n_tt') * link_fdr >= td_prob,
        link_dd_check = pl.col('n_dd') * td_dd_ratio <= pl.col('n_td'),
    )

    for row in df_link_checks.to_dicts():
        fdr_group = row['fdr_group']
        td_check = row['link_td_check']
        dd_check = row['link_dd_check']
        if not td_check:
            warnings.warn(f'Insufficient TT for link FDR in group {fdr_group}.')
        if not dd_check:
            warnings.warn(f'More DD than TT for link FDR in group {fdr_group}.')

    df_link = df_link_checks.explode(
        pl.selectors.list()
    ).select(
        *df_link.columns,
        'link_td_check',
        'link_dd_check',
    )

    return df_link


def _ppi_fdr(df_link, agg, ppi_fdr, first_aggs, never_agg_cols, td_prob, td_dd_ratio, window_size: int = 5):
    ppi_merge_cols = [c for c in df_link.columns if c not in ppi_cols+never_agg_cols]
    df_ppi = df_link.group_by(ppi_cols).agg(
        *first_aggs,
        *[
            pl.col(c).list.explode()
            for c in ppi_merge_cols
        ],
        score=agg
    )
    df_ppi = df_ppi.with_columns(
        ppi_fdr = single_grouped_fdr(df_ppi, window_size=window_size)
    ).filter(pl.col('ppi_fdr').clip(0.0, 1.0) <= ppi_fdr)

    df_ppi_checks = df_ppi.group_by('fdr_group').agg(
        pl.col('TD').sum().alias('n_td'),
        pl.col('DD').sum().alias('n_dd'),
        pl.col('TT').sum().alias('n_tt'),
        pl.all(),
    ).with_columns(
        ppi_td_check = pl.col('n_tt') * ppi_fdr >= td_prob,
        ppi_dd_check = pl.col('n_dd') * td_dd_ratio <= pl.col('n_td'),
    )

    for row in df_ppi_checks.to_dicts():
        fdr_group = row['fdr_group']
        td_check = row['ppi_td_check']
        dd_check = row['ppi_dd_check']
        if not td_check:
            warnings.warn(f'Insufficient TT for PPI FDR in group {fdr_group}.')
        if not dd_check:
            warnings.warn(f'More DD than TT for PPI FDR in group {fdr_group}.')

    df_ppi = df_ppi_checks.explode(
        pl.selectors.list()
    ).select(
        *df_ppi.columns,
        'ppi_td_check',
        'ppi_dd_check',
    )

    return df_ppi


def single_grouped_fdr(df: Union[pl.DataFrame, pd.DataFrame],
                       fdr_group_col: str = "fdr_group",
                       unpaired_groups: list[str] = None,
                       window_size: int = 5) -> pl.Series:
    """
    Computes the false discovery rate (FDR) for a given DF.

    Parameters
    ----------
    df : pl.DataFrame|pd.DataFrame
        The input DF containing columns for TT, TD, DD, decoy_class and score.
    fdr_group_col : str
        The column name for grouping
    unpaired_groups : list[str], optional
        Groups where TD matches are absent and DD matches should be treated as TD.
    window_size : int
        Number of TD matches per sliding window for interpolated decoy FDR estimation.

    Returns
    -------
    pl.Series
        A polars series containing the FDR for each row of the input.
    """
    if not isinstance(df, pl.DataFrame):
        df: pl.DataFrame = pl.DataFrame(df)

    order_col = 'order_col'
    while order_col in df.columns:
        order_col += '_'

    df = df.with_row_index(order_col)
    fdr_with_order = pl.DataFrame(
        schema={**df.schema, **{'fdr': pl.Float64}}
    )
    fdr_groups = df[fdr_group_col].unique().to_list()
    for fdr_group in fdr_groups:
        class_df = df.filter(
            pl.col(fdr_group_col) == fdr_group
        )
        if unpaired_groups and fdr_group in unpaired_groups:
            # Unpaird groups don't have proper TD matched
            class_df = class_df.with_columns(
                TD = 'DD',
                DD = pl.lit(False)
            )
        class_df = class_df.with_columns(
            single_fdr(class_df, window_size=window_size)
        )
        fdr_with_order = fdr_with_order.extend(class_df)

    return fdr_with_order.sort(order_col)['fdr']


def _clipped_window_fdr(
    w_tt: np.ndarray, w_td: np.ndarray, w_dd: np.ndarray
) -> np.ndarray:
    """Compute clipped (TD - DD) / TT for each window, handling TT == 0 safely."""
    w_fp = w_td - w_dd
    safe_tt = np.where(w_tt > 0, w_tt, 1)
    return np.where(
        w_tt > 0,
        np.clip(w_fp / safe_tt, 0.0, 1.0),
        np.where(w_fp > 0, 1.0, 0.0),
    )


def single_fdr(df: Union[pl.DataFrame, pd.DataFrame], window_size: int = 5) -> pl.Series:
    """
    Compute monotonic global q-values using a sliding-window interpolated decoy local FDR.

    Matches are ordered from lowest to highest score. Windows of `window_size` TD
    matches slide one TD at a time, estimating each window's local false-positive
    ratio `(TD - DD) / TT` clipped to `[0.0, 1.0]`. Overlapping window values per
    match are averaged, the post-last-TD tail extends from the last window's start
    to each trailing match, and local FDRs on TT matches are integrated cumulatively
    from highest to lowest score and made monotonic.

    Parameters
    ----------
    df : pl.DataFrame|pd.DataFrame
        Input DataFrame containing `TT`, `TD`, `DD`, and `score` columns.
    window_size : int
        Number of TD matches in each sliding window (default 5).

    Returns
    -------
    pl.Series
        A Float64 polars Series named `'fdr'` aligned to the input row order.
    """
    if not isinstance(df, pl.DataFrame):
        df = pl.DataFrame(df)
    if window_size < 1:
        raise ValueError("window_size must be at least 1")

    n = df.height
    if n == 0:
        return pl.Series('fdr', [], dtype=pl.Float64)

    order_col = 'order_col'
    while order_col in df.columns:
        order_col += '_'

    working_df = (
        df.select(['TT', 'TD', 'DD', 'score'])
        .with_row_index(order_col)
        .sort('score', descending=False, maintain_order=True)
    )
    tt = working_df['TT'].cast(pl.Boolean).fill_null(False).to_numpy().astype(np.int64)
    td = working_df['TD'].cast(pl.Boolean).fill_null(False).to_numpy().astype(np.int64)
    dd = working_df['DD'].cast(pl.Boolean).fill_null(False).to_numpy().astype(np.int64)

    td_idx = np.flatnonzero(td)
    m = len(td_idx)
    if m == 0:
        fdr_asc = np.zeros(n, dtype=np.float64)
    else:
        pref_tt = np.empty(n + 1, dtype=np.int64)
        pref_tt[0] = 0
        np.cumsum(tt, out=pref_tt[1:])
        pref_td = np.empty(n + 1, dtype=np.int64)
        pref_td[0] = 0
        np.cumsum(td, out=pref_td[1:])
        pref_dd = np.empty(n + 1, dtype=np.int64)
        pref_dd[0] = 0
        np.cumsum(dd, out=pref_dd[1:])

        eff_w = min(window_size, m)
        k = m - eff_w + 1
        starts = np.empty(k, dtype=np.int64)
        starts[0] = 0
        starts[1:] = td_idx[:m - eff_w] + 1
        ends = td_idx[eff_w - 1:]

        w_val = _clipped_window_fdr(
            pref_tt[ends + 1] - pref_tt[starts],
            pref_td[ends + 1] - pref_td[starts],
            pref_dd[ends + 1] - pref_dd[starts],
        )

        # Average overlapping windows across the m segments [0..t_0], ..., [t_{m-2}+1..t_{m-1}]
        seg_idx = np.arange(m, dtype=np.int64)
        k_lo = np.maximum(0, seg_idx - eff_w + 1)
        k_hi = np.minimum(k - 1, seg_idx)
        pref_w = np.empty(k + 1, dtype=np.float64)
        pref_w[0] = 0.0
        np.cumsum(w_val, out=pref_w[1:])
        seg_mean = np.clip((pref_w[k_hi + 1] - pref_w[k_lo]) / (k_hi - k_lo + 1), 0.0, 1.0)

        seg_lens = np.empty(m, dtype=np.int64)
        seg_lens[0] = td_idx[0] + 1
        seg_lens[1:] = np.diff(td_idx)

        pep = np.empty(n, dtype=np.float64)
        last_td = int(td_idx[-1])
        pep[:last_td + 1] = np.repeat(seg_mean, seg_lens)

        if last_td + 1 < n:
            tail_start = int(starts[-1])
            tail_ends = np.arange(last_td + 1, n, dtype=np.int64)
            pep[last_td + 1:] = _clipped_window_fdr(
                pref_tt[tail_ends + 1] - pref_tt[tail_start],
                pref_td[tail_ends + 1] - pref_td[tail_start],
                pref_dd[tail_ends + 1] - pref_dd[tail_start],
            )

        # Integrate local FDR cumulatively from highest score down to lowest score
        cum_tt_desc = np.cumsum(tt[::-1])
        cum_fp_desc = np.cumsum((pep * tt)[::-1])
        if cum_tt_desc[-1] == 0:
            fdr_asc = np.minimum.accumulate(pep)
        else:
            safe_cum_tt = np.where(cum_tt_desc > 0, cum_tt_desc, 1)
            global_fdr_desc = np.where(cum_tt_desc > 0, cum_fp_desc / safe_cum_tt, np.inf)
            fdr_asc = np.minimum.accumulate(global_fdr_desc[::-1])

    working_df = working_df.with_columns(pl.Series('fdr', fdr_asc, dtype=pl.Float64))
    return working_df.sort(order_col)['fdr']
