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

    df_csm = _csm_fdr(df, csm_fdr, unique_csm, td_prob, td_dd_ratio)

    # Calculate peptide FDR and filter
    logger.debug('Calculate peptide FDR and filter')
    df_pep = _pep_fdr(df_csm, aggs['pep'], pep_fdr, first_aggs, never_agg_cols, td_prob, td_dd_ratio)

    logger.debug('Calculate protein FDR and filter')
    df_prot = _prot_fdr(df_pep, aggs['prot'], prot_fdr, td_prot_prob)

    logger.debug('Filter peptide pairs for passed proteins')
    df_pep = _prot_filter(df_pep, df_prot, decoy_adjunct)

    # Calculate link FDR and cutoff
    logger.debug('Calculate link FDR and cutoff')
    df_link = _link_fdr(df_pep, aggs['link'], link_fdr, first_aggs, never_agg_cols, td_prob, td_dd_ratio)

    # Calculate PPI FDR
    logger.debug('Calculate PPI FDR')
    df_ppi = _ppi_fdr(df_link, aggs['prot'], ppi_fdr, first_aggs, never_agg_cols, td_prob, td_dd_ratio)

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

def _csm_fdr(df, csm_fdr, unique_csm, td_prob, td_dd_ratio):
    if unique_csm:
        df_csm = df.sort('score', descending=True).unique(subset=csm_cols, keep='first')
    else:
        df_csm = df

    # Calculate CSM FDR and cutoff
    logger.debug('Calculate CSM FDR and cutoff')
    df_csm = df_csm.with_columns(
        csm_fdr = single_grouped_fdr(df_csm)
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


def _pep_fdr(df_csm, agg, pep_fdr, first_aggs, never_agg_cols, td_prob, td_dd_ratio):
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
        pep_fdr = single_grouped_fdr(df_pep)
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
              td_prot_prob) -> pl.DataFrame:
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
        prot_fdr=single_grouped_fdr(df_prot, fdr_group_col='protein_fdr_group')
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


def _link_fdr(df_pep, agg, link_fdr, first_aggs, never_agg_cols, td_prob, td_dd_ratio):
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
        link_fdr = single_grouped_fdr(df_link)
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


def _ppi_fdr(df_link, agg, ppi_fdr, first_aggs, never_agg_cols, td_prob, td_dd_ratio):
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
        ppi_fdr = single_grouped_fdr(df_ppi)
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
                       unpaired_groups: list[str] = None) -> pl.Series:
    """
    Computes the false discovery rate (FDR) for a given DF.

    Parameters
    ----------
    df : pl.DataFrame|pd.DataFrame
        The input DF containing columns for TT, TD, DD, decoy_class and score.
    fdr_group_col : str
        The column name for grouping

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
        schema={**df.schema, **{'fdr': pl.Float32}}
    )
    fdr_with_order = fdr_with_order.with_columns(
        fdr = pl.lit(0.0)
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
            single_fdr(class_df)
        )
        fdr_with_order = fdr_with_order.extend(class_df)

    return fdr_with_order.sort(order_col)['fdr']


def single_fdr(df: Union[pl.DataFrame, pd.DataFrame]) -> pl.Series:
    working_df = df.select([
        'TT',
        'TD',
        'DD',
        'score'
    ])
    order_col = 'order_col'
    while order_col in df.columns:
        order_col += '_'

    working_df = working_df.with_row_index(order_col)
    working_df = working_df.sort('score', descending=True)
    fdr_raw = (
        (working_df['TD'].cast(pl.Int32).cum_sum() - working_df['DD'].cast(pl.Int32).cum_sum())
        / working_df['TT'].cast(pl.Int32).cum_sum()
    )
    working_df = working_df.with_columns(
        fdr = fdr_raw.clip(lower_bound=0).reverse().cum_min().reverse()
    )
    return working_df.sort(order_col)['fdr']


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
        df_filtered = input_df
        col_levels = cutoffs[5:]
        neg_col_levels = col_levels[len(boost_cols):]

        for i, c in enumerate(boost_cols):
            df_filtered = df_filtered.filter(
                (
                    (pl.col(c) - pl.col(c).min()) /
                    (pl.col(c).max() - pl.col(c).min())
                ) >= col_levels[i]
            )

        for i, c in enumerate(neg_boost_cols):
            df_filtered = df_filtered.filter(
                (
                    (pl.col(c) - pl.col(c).min()) /
                    (pl.col(c).max() - pl.col(c).min())
                ) <= neg_col_levels[i]
            )
            
        return full_fdr(
            df=df_filtered,
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
