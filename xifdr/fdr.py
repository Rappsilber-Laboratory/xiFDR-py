import logging
import warnings
from typing import Union

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


def _get_columns(df: Union[pl.DataFrame, pl.LazyFrame]) -> list[str]:
    if isinstance(df, pl.LazyFrame):
        return df.collect_schema().names()
    return df.columns


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
             custom_aggs:dict = None) -> dict[str, pl.DataFrame]:
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
    df_l = df.lazy().filter(
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
    df_cols = _get_columns(df_l)
    missing_columns = [
        c for c in required_columns
        if c not in df_cols
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

    df_csm_l = _csm_fdr(df_l, csm_fdr, unique_csm, td_prob, td_dd_ratio).collect().lazy()

    # Calculate peptide FDR and filter
    logger.debug('Calculate peptide FDR and filter')
    df_pep_l = _pep_fdr(df_csm_l, aggs['pep'], pep_fdr, first_aggs, never_agg_cols, td_prob, td_dd_ratio).collect().lazy()

    logger.debug('Calculate protein FDR and filter')
    df_prot_l = _prot_fdr(df_pep_l, aggs['prot'], prot_fdr, td_prot_prob).collect().lazy()

    logger.debug('Filter peptide pairs for passed proteins')
    df_pep_l = _prot_filter(df_pep_l, df_prot_l, decoy_adjunct).collect().lazy()

    # Calculate link FDR and cutoff
    logger.debug('Calculate link FDR and cutoff')
    df_link_l = _link_fdr(df_pep_l, aggs['link'], link_fdr, first_aggs, never_agg_cols, td_prob, td_dd_ratio).collect().lazy()

    # Calculate PPI FDR
    logger.debug('Calculate PPI FDR')
    df_ppi_l = _ppi_fdr(df_link_l, aggs['prot'], ppi_fdr, first_aggs, never_agg_cols, td_prob, td_dd_ratio).collect().lazy()

    # Back-fitler levels
    df_ppi_l = df_ppi_l.with_columns(pass_threshold=pl.lit(True))

    pass_on_cols = [pl.col('score').alias('ppi_score'), 'ppi_fdr', 'pass_threshold']
    df_link_l = df_link_l.join(
        df_ppi_l.select(*ppi_cols, *pass_on_cols),
        on=ppi_cols,
        how='full',
        coalesce=True,
    ).with_columns(
        pass_threshold=pl.col('pass_threshold').fill_null(pl.lit(False))
    )

    pass_on_cols += [pl.col('score').alias('link_score'), 'link_fdr']
    df_pep_l = df_pep_l.join(
        df_link_l.select(*link_cols, *pass_on_cols),
        on=link_cols,
        how='full',
        coalesce=True,
    ).with_columns(
        pass_threshold=pl.col('pass_threshold').fill_null(pl.lit(False))
    )

    pass_on_cols += [pl.col('score').alias('pep_score'), 'pep_fdr']
    df_csm_l = df_csm_l.join(
        df_pep_l.select(*pep_cols, *pass_on_cols),
        on=pep_cols,
        how='full',
        coalesce=True,
    ).with_columns(
        pass_threshold=pl.col('pass_threshold').fill_null(pl.lit(False))
    )

    if filter_back:
        df_link_l = df_link_l.filter('pass_threshold')
        df_pep_l = df_pep_l.filter('pass_threshold')
        df_csm_l = df_csm_l.filter('pass_threshold')

    df_ppi_l = df_ppi_l.with_columns(
        protein_p1='ppi_protein_p1',
        protein_p2='ppi_protein_p2'
    )

    collected = pl.collect_all([df_csm_l, df_pep_l, df_prot_l, df_link_l, df_ppi_l])
    res = {
        'csm': collected[0],
        'pep': collected[1],
        'prot': collected[2],
        'link': collected[3],
        'ppi': collected[4],
    }

    for level, df_col in res.items():
        if level == 'prot':
            group_col = 'protein_fdr_group'
            level_name = 'protein'
        else:
            group_col = 'fdr_group'
            level_name = {
                'csm': 'CSM',
                'pep': 'peptide',
                'link': 'link',
                'ppi': 'PPI'
            }[level]

        td_col = f'{level}_td_check'
        dd_col = f'{level}_dd_check'
        
        check_cols = [group_col]
        if td_col in df_col.columns:
            check_cols.append(td_col)
        if dd_col in df_col.columns:
            check_cols.append(dd_col)
            
        if len(check_cols) > 1:
            checks = df_col.select(check_cols).unique()
            for row in checks.to_dicts():
                fdr_group = row[group_col]
                if td_col in row and not row[td_col]:
                    warnings.warn(f'Insufficient TT for {level_name} FDR in group {fdr_group}.')
                if dd_col in row and not row[dd_col]:
                    warnings.warn(f'More DD than TT for {level_name} FDR in group {fdr_group}.')

    return res

def _csm_fdr(df_l, csm_fdr, unique_csm, td_prob, td_dd_ratio):
    is_lazy = isinstance(df_l, pl.LazyFrame)
    if not is_lazy:
        df_l = df_l.lazy()

    if unique_csm:
        df_csm_l = df_l.sort('score', descending=True).unique(subset=csm_cols, keep='first')
    else:
        df_csm_l = df_l

    # Calculate CSM FDR and cutoff
    logger.debug('Calculate CSM FDR and cutoff')
    df_csm_l = df_csm_l.sort('score', descending=True).with_columns(
        csm_fdr = single_grouped_fdr()
    ).filter(pl.col('csm_fdr').clip(0.0, 1.0) <= csm_fdr)

    df_csm_checks = df_csm_l.group_by('fdr_group').agg(
        pl.col('TD').sum().alias('n_td'),
        pl.col('DD').sum().alias('n_dd'),
        pl.col('TT').sum().alias('n_tt'),
        pl.all(),
    ).with_columns(
        csm_td_check = pl.col('n_tt') * csm_fdr >= td_prob,
        csm_dd_check = pl.col('n_dd') * td_dd_ratio <= pl.col('n_td'),
    )

    csm_cols_list = _get_columns(df_csm_l)
    df_csm_l = df_csm_checks.explode(
        pl.selectors.list()
    ).select(
        *csm_cols_list,
        'csm_td_check',
        'csm_dd_check',
    )

    return df_csm_l if is_lazy else df_csm_l.collect()


def _pep_fdr(df_csm_l, agg, pep_fdr, first_aggs, never_agg_cols, td_prob, td_dd_ratio):
    is_lazy = isinstance(df_csm_l, pl.LazyFrame)
    if not is_lazy:
        df_csm_l = df_csm_l.lazy()

    csm_cols_list = _get_columns(df_csm_l)
    pep_merge_cols = [c for c in csm_cols_list if c not in pep_cols+never_agg_cols]
    df_pep_l = df_csm_l.group_by(pep_cols).agg(
        *first_aggs,
        *[
            pl.col(c).list.explode()
            for c in pep_merge_cols
        ],
        protein_score_p1=expression_utils.replace_input(agg, 'protein_score_p1'),
        protein_score_p2=expression_utils.replace_input(agg, 'protein_score_p2'),
        score=agg
    )
    df_pep_l = df_pep_l.sort('score', descending=True).with_columns(
        pep_fdr = single_grouped_fdr()
    ).filter(pl.col('pep_fdr').clip(0.0, 1.0) <= pep_fdr)

    df_pep_checks = df_pep_l.group_by('fdr_group').agg(
        pl.col('TD').sum().alias('n_td'),
        pl.col('DD').sum().alias('n_dd'),
        pl.col('TT').sum().alias('n_tt'),
        pl.all(),
    ).with_columns(
        pep_td_check = pl.col('n_tt') * pep_fdr >= td_prob,
        pep_dd_check = pl.col('n_dd') * td_dd_ratio <= pl.col('n_td'),
    )

    pep_cols_list = _get_columns(df_pep_l)
    df_pep_l = df_pep_checks.explode(
        pl.selectors.list()
    ).select(
        *pep_cols_list,
        'pep_td_check',
        'pep_dd_check',
    )

    return df_pep_l if is_lazy else df_pep_l.collect()


def _prot_fdr(df_pep_l:Union[pl.DataFrame, pl.LazyFrame],
              agg,
              prot_fdr,
              td_prot_prob) -> Union[pl.DataFrame, pl.LazyFrame]:
    is_lazy = isinstance(df_pep_l, pl.LazyFrame)
    if not is_lazy:
        df_pep_l = df_pep_l.lazy()

    # Construct protein (group) DF
    df_prot_p1_l = df_pep_l.select([
        'protein_p1', 'protein_score_p1', 'decoy_p1', 'fdr_group'
    ]).rename({
        'protein_p1': 'protein',
        'protein_score_p1': 'score',
        'decoy_p1': 'decoy',
    })

    df_prot_p2_l = df_pep_l.select([
        'protein_p2', 'protein_score_p2', 'decoy_p2', 'fdr_group'
    ]).rename({
        'protein_p2': 'protein',
        'protein_score_p2': 'score',
        'decoy_p2': 'decoy',
    })

    df_prot_l = pl.concat([
        df_prot_p1_l,
        df_prot_p2_l
    ])
    df_prot_l = df_prot_l.with_columns(
        protein_group=pl.col('protein').list.unique().list.sort()
    )
    df_prot_l = df_prot_l.group_by(['protein_group', 'decoy']).agg(
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
    ).with_columns(
        TD=pl.col('decoy'),
        TT=~pl.col('decoy'),
        DD=pl.lit(False)  # Abuse CL-FDR for linear case
    )
    df_prot_l = df_prot_l.sort('score', descending=True).with_columns(
        prot_fdr=single_grouped_fdr(fdr_group_col='protein_fdr_group')
    )
    df_prot_l = df_prot_l.filter(pl.col('prot_fdr').clip(0.0, 1.0) <= prot_fdr)

    # Check whether there are at least enough TT to have approx. `min_td` TD matches under the requested FDR level.
    df_prot_checks = df_prot_l.group_by('protein_fdr_group').agg(
        pl.col('TT').sum().alias('n_t'),
        pl.all(),
    ).with_columns(
        prot_td_check = pl.col('n_t') * prot_fdr >= td_prot_prob,
    )

    prot_cols_list = _get_columns(df_prot_l)
    df_prot_l = df_prot_checks.explode(
        pl.selectors.list()
    ).select(
        *prot_cols_list,
        'prot_td_check',
    )

    return df_prot_l if is_lazy else df_prot_l.collect()


def _prot_filter(df_pep_l, df_prot_l, decoy_adjunct):
    is_lazy = isinstance(df_pep_l, pl.LazyFrame)
    if isinstance(df_prot_l, pl.LazyFrame):
        passed_prots = (
            df_prot_l.select('protein')
            .explode('protein')
            .select(
                pl.col('protein')
                .list.join(';')
                .str.replace_all(decoy_adjunct, '')
                .str.split(';')
                .list.sort()
                .list.join(';')
                .alias('passed_prots')
            )
            .unique()
            .collect()
        )
    else:
        passed_prots = (
            df_prot_l.select('protein')
            .explode('protein')
            .select(
                pl.col('protein')
                .list.join(';')
                .str.replace_all(decoy_adjunct, '')
                .str.split(';')
                .list.sort()
                .list.join(';')
                .alias('passed_prots')
            )
            .unique()
        )

    passed_prots_join = passed_prots.lazy() if is_lazy else passed_prots

    ## Filter left over peptide pairs
    logger.debug('Filter left over peptide pairs')
    df_pep_l = df_pep_l.with_columns(
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

    return df_pep_l.join(
        passed_prots_join,
        left_on=['base_protein_p1'],
        right_on=['passed_prots'],
        how='inner',
        suffix='p1'
    ).join(
        passed_prots_join,
        left_on=['base_protein_p2'],
        right_on=['passed_prots'],
        how='inner',
        suffix='p2'
    )


def _link_fdr(df_pep_l, agg, link_fdr, first_aggs, never_agg_cols, td_prob, td_dd_ratio):
    is_lazy = isinstance(df_pep_l, pl.LazyFrame)
    if not is_lazy:
        df_pep_l = df_pep_l.lazy()

    pep_cols_list = _get_columns(df_pep_l)
    link_merge_cols = [c for c in pep_cols_list if c not in link_cols+never_agg_cols]
    df_link_l = df_pep_l.filter(
        pl.col('fdr_group') != "linear" # Disregard linear peptides from here on
    ).group_by(link_cols).agg(
        *first_aggs,
        *[
            pl.col(c).list.explode()
            for c in link_merge_cols
        ],
        score=agg
    )
    df_link_l = df_link_l.sort('score', descending=True).with_columns(
        link_fdr = single_grouped_fdr()
    ).filter(pl.col('link_fdr').clip(0.0, 1.0) <= link_fdr)

    df_link_checks = df_link_l.group_by('fdr_group').agg(
        pl.col('TD').sum().alias('n_td'),
        pl.col('DD').sum().alias('n_dd'),
        pl.col('TT').sum().alias('n_tt'),
        pl.all(),
    ).with_columns(
        link_td_check = pl.col('n_tt') * link_fdr >= td_prob,
        link_dd_check = pl.col('n_dd') * td_dd_ratio <= pl.col('n_td'),
    )

    link_cols_list = _get_columns(df_link_l)
    df_link_l = df_link_checks.explode(
        pl.selectors.list()
    ).select(
        *link_cols_list,
        'link_td_check',
        'link_dd_check',
    )

    return df_link_l if is_lazy else df_link_l.collect()


def _ppi_fdr(df_link_l, agg, ppi_fdr, first_aggs, never_agg_cols, td_prob, td_dd_ratio):
    is_lazy = isinstance(df_link_l, pl.LazyFrame)
    if not is_lazy:
        df_link_l = df_link_l.lazy()

    link_cols_list = _get_columns(df_link_l)
    ppi_merge_cols = [c for c in link_cols_list if c not in ppi_cols+never_agg_cols]
    df_ppi_l = df_link_l.group_by(ppi_cols).agg(
        *first_aggs,
        *[
            pl.col(c).list.explode()
            for c in ppi_merge_cols
        ],
        score=agg
    )
    df_ppi_l = df_ppi_l.sort('score', descending=True).with_columns(
        ppi_fdr = single_grouped_fdr()
    ).filter(pl.col('ppi_fdr').clip(0.0, 1.0) <= ppi_fdr)

    df_ppi_checks = df_ppi_l.group_by('fdr_group').agg(
        pl.col('TD').sum().alias('n_td'),
        pl.col('DD').sum().alias('n_dd'),
        pl.col('TT').sum().alias('n_tt'),
        pl.all(),
    ).with_columns(
        ppi_td_check = pl.col('n_tt') * ppi_fdr >= td_prob,
        ppi_dd_check = pl.col('n_dd') * td_dd_ratio <= pl.col('n_td'),
    )

    ppi_cols_list = _get_columns(df_ppi_l)
    df_ppi_l = df_ppi_checks.explode(
        pl.selectors.list()
    ).select(
        *ppi_cols_list,
        'ppi_td_check',
        'ppi_dd_check',
    )

    return df_ppi_l if is_lazy else df_ppi_l.collect()


def single_grouped_fdr(fdr_group_col: str = "fdr_group",
                       unpaired_groups: list[str] = None) -> pl.Expr:
    """
    Computes the false discovery rate (FDR) expression for a given group.
    Assumes the DataFrame will be sorted by score descending before evaluating.

    Parameters
    ----------
    fdr_group_col : str
        The column name for grouping
    unpaired_groups : list[str]
        List of groups to treat as unpaired (TD = DD, DD = 0)

    Returns
    -------
    pl.Expr
        A polars expression computing the FDR.
    """
    td_col = pl.col('TD').cast(pl.Int32)
    dd_col = pl.col('DD').cast(pl.Int32)
    
    if unpaired_groups:
        is_unpaired = pl.col(fdr_group_col).is_in(unpaired_groups)
        td_col = pl.when(is_unpaired).then(pl.col('DD').cast(pl.Int32)).otherwise(td_col)
        dd_col = pl.when(is_unpaired).then(pl.lit(0)).otherwise(dd_col)
        
    td_cum = td_col.cum_sum().over(fdr_group_col)
    dd_cum = dd_col.cum_sum().over(fdr_group_col)
    tt_cum = pl.col('TT').cast(pl.Int32).cum_sum().over(fdr_group_col)
    
    fdr_raw = (td_cum - dd_cum) / tt_cum
    return fdr_raw.clip(lower_bound=0).reverse().cum_min().reverse().over(fdr_group_col)


def single_fdr() -> pl.Expr:
    """
    Computes the single false discovery rate (FDR) expression.
    Assumes the DataFrame will be sorted by score descending before evaluating.

    Returns
    -------
    pl.Expr
        A polars expression computing the FDR.
    """
    td_cum = pl.col('TD').cast(pl.Int32).cum_sum()
    dd_cum = pl.col('DD').cast(pl.Int32).cum_sum()
    tt_cum = pl.col('TT').cast(pl.Int32).cum_sum()
    
    fdr_raw = (td_cum - dd_cum) / tt_cum
    return fdr_raw.clip(lower_bound=0).reverse().cum_min().reverse()


def group_full_fdr(df: Union[pl.DataFrame, pd.DataFrame],
                   cutoffs_self: list[float],
                   cutoffs_between: list[float],
                   boost_cols: list[str] = None,
                   neg_boost_cols: list[str] = None,
                   min_len: int = 5,
                   decoy_adjunct: str = 'REV_',
                   unique_csm: bool = True,
                   filter_back: bool = True,
                   prepare_column: bool = True,
                   td_prob: int = 2,
                   td_prot_prob: int = 10,
                   td_dd_ratio: float = 1.0,
                   custom_aggs: dict = None) -> dict[str, pl.DataFrame]:
    """
    Apply two sets of FDR cutoffs (e.g. from group_boost) for 'self' and 'between' matches and merge.

    Parameters
    ----------
    df
        Input CSM dataframe
    cutoffs_self
        Cutoffs generated by boost() for the 'self' group
    cutoffs_between
        Cutoffs generated by boost() for the 'between' group
    boost_cols
        Columns in which to look for lower cutoffs (used in boosting)
    neg_boost_cols
        Columns in which to look for upper cutoffs (used in boosting)
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

    Returns
    -------
        Return a dict with keys `'csm'`, `'pep'`, `'prot'`, `'link'`, `'ppi'` that contains the resulting polars DataFrame for each FDR level.
    """
    if boost_cols is None:
        boost_cols = []
    if neg_boost_cols is None:
        neg_boost_cols = []

    if prepare_column:
        df = prepare_columns(df, decoy_adjunct=decoy_adjunct)
        # Avoid running it again in full_fdr
        prepare_column = False

    def _apply_cutoffs(input_df, cutoffs):
        df_filtered_l = input_df.lazy()
        col_levels = cutoffs[5:]
        neg_col_levels = col_levels[len(boost_cols):]

        for i, c in enumerate(boost_cols):
            df_filtered_l = df_filtered_l.filter(
                (
                    (pl.col(c) - pl.col(c).min()) /
                    (pl.col(c).max() - pl.col(c).min())
                ) >= col_levels[i]
            )

        for i, c in enumerate(neg_boost_cols):
            df_filtered_l = df_filtered_l.filter(
                (
                    (pl.col(c) - pl.col(c).min()) /
                    (pl.col(c).max() - pl.col(c).min())
                ) <= neg_col_levels[i]
            )
            
        return full_fdr(
            df=df_filtered_l.collect(),
            csm_fdr=cutoffs[0],
            pep_fdr=cutoffs[1],
            prot_fdr=cutoffs[2],
            link_fdr=cutoffs[3],
            ppi_fdr=cutoffs[4],
            min_len=min_len,
            decoy_adjunct=decoy_adjunct,
            unique_csm=unique_csm,
            filter_back=filter_back,
            prepare_column=prepare_column,
            td_prob=td_prob,
            td_prot_prob=td_prot_prob,
            td_dd_ratio=td_dd_ratio,
            custom_aggs=custom_aggs
        )

    logger.info("Applying full_fdr for 'self' cutoffs")
    res_self = _apply_cutoffs(df, cutoffs_self)
    
    logger.info("Applying full_fdr for 'between' cutoffs")
    res_between = _apply_cutoffs(df, cutoffs_between)
    
    merged = {}
    
    for level in ['csm', 'pep', 'link', 'ppi']:
        df_self = res_self[level].filter(pl.col('fdr_group').is_in(['self', 'linear', 'overlapping']))
        df_between = res_between[level].filter(pl.col('fdr_group') == 'between')
        merged[level] = pl.concat([df_self, df_between], how="vertical")
        
    prot_self = res_self['prot']
    prot_between = res_between['prot']
    
    # Identify join keys for proteins
    # The protein level grouping uses 'protein_group' and 'decoy'
    join_keys = ['protein_group', 'decoy']
    
    merged['prot'] = prot_self.join(
        prot_between,
        on=join_keys,
        how='full',
        suffix='_between'
    )
    # Rename columns from left side to _self
    rename_cols = {
        col: f"{col}_self" 
        for col in prot_self.columns 
        if col not in join_keys and f"{col}_between" in merged['prot'].columns
    }
    merged['prot'] = merged['prot'].rename(rename_cols)
    
    return merged
