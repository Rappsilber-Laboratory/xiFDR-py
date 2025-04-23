import typing
import logging
import pandas as pd
import polars as pl
from xifdr.utils.column_preparation import prepare_columns
from xifdr.utils import expression_utils


logger = logging.getLogger(__name__)

def full_fdr(df: pl.DataFrame | pd.DataFrame,
             csm_fdr:float = 1.0,
             pep_fdr:float = 1.0,
             prot_fdr:float = 1.0,
             link_fdr:float = 1.0,
             ppi_fdr:float = 1.0,
             decoy_adjunct:str = 'REV_',
             unique_csm: bool = True,
             filter_back:bool = True,
             prepare_column:bool = True,
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
    decoy_adjunct
        Prefix/Suffix indicating a decoy match
    unique_csm
        Make CSMs unique
    filter_back
        Filter lower levels to include only matches that also pass on higher levels
    prepare_column
        Perform preparation of aggregation columns like sorting ambiguous proteins and swapping protein 1/2
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

    csm_cols = required_columns.copy()
    csm_cols.remove('score')
    csm_cols.remove('start_pos_p1')
    csm_cols.remove('start_pos_p2')
    csm_cols.remove('link_pos_p1')
    csm_cols.remove('link_pos_p2')
    csm_cols += ['cl_pos_p1', 'cl_pos_p2']

    if unique_csm:
        df_csm = df.sort('score', descending=True).unique(subset=csm_cols, keep='first')
    else:
        df_csm = df

    # Calculate CSM FDR and cutoff
    logger.debug('Calculate CSM FDR and cutoff')
    df_csm = df_csm.with_columns(
        csm_fdr = single_grouped_fdr(df_csm)
    )
    df_csm = df_csm.filter(pl.col('csm_fdr') <= csm_fdr)

    # Calculate peptide FDR and filter
    logger.debug('Calculate peptide FDR and filter')
    pep_cols = csm_cols.copy()
    pep_cols.remove('charge')
    pep_merge_cols = [c for c in df_csm.columns if c not in pep_cols+never_agg_cols]
    df_pep = df_csm.group_by(pep_cols).agg(
        *first_aggs,
        *[
            pl.col(c).flatten()
            for c in pep_merge_cols
        ],
        protein_score_p1=expression_utils.replace_input(aggs['pep'], 'protein_score_p1'),
        protein_score_p2=expression_utils.replace_input(aggs['pep'], 'protein_score_p2'),
        score=aggs['pep']
    )
    df_pep = df_pep.with_columns(
        pep_fdr = single_grouped_fdr(df_pep)
    )
    df_pep = df_pep.filter(pl.col('pep_fdr') <= pep_fdr)

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

    ## Calculate and filter protein group FDR as linear FDR
    logger.debug('Calculate and filter protein group FDR as linear FDR')
    df_prot = pl.concat([
        df_prot_p1,
        df_prot_p2
    ])
    df_prot = df_prot.with_columns(
        protein_group = pl.col('protein').list.unique()
    )
    df_prot = df_prot.group_by(['protein_group', 'decoy']).agg(
        pl.col('protein'),
        pl.col('fdr_group'),
        score=aggs['prot']
    ).with_columns(
        no_self = ~pl.lit('self').is_in(pl.col('fdr_group')),
        no_linear = ~pl.lit('linear').is_in(pl.col('fdr_group')),
        between = pl.lit('between').is_in(pl.col('fdr_group')),
    ).with_columns(
        protein_fdr_group = (
            pl.when(pl.col('between') & pl.col('no_self') & pl.col('no_linear'))
              .then(pl.lit('unsupported_between'))
              .when(pl.col('between'))
              .then(pl.lit('supported_between'))
              .otherwise(pl.lit('self_or_linear'))
        )
    )
    df_prot = df_prot.with_columns(
        TD = pl.col('decoy'),
        TT = ~pl.col('decoy'),
        DD = pl.lit(False)  # Abuse CL-FDR for linear case
    )
    df_prot = df_prot.with_columns(
        prot_fdr = single_grouped_fdr(df_prot, fdr_group_col='protein_fdr_group')
    )

    passed_prots = df_prot.filter(pl.col('prot_fdr') <= prot_fdr)['protein'].explode()
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

    df_pep = df_pep.join(
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

    # Calculate link FDR and cutoff
    logger.debug('Calculate link FDR and cutoff')
    link_cols = pep_cols.copy()
    link_cols.remove('sequence_p1')
    link_cols.remove('sequence_p2')
    link_merge_cols = [c for c in df_pep.columns if c not in link_cols+never_agg_cols]
    df_link = df_pep.filter(
        pl.col('fdr_group') != "linear" # Disregard linear peptides from here on
    ).group_by(link_cols).agg(
        *first_aggs,
        *[
            pl.col(c).flatten()
            for c in link_merge_cols
        ],
        score=aggs['link']
    )
    df_link = df_link.with_columns(
        link_fdr = single_grouped_fdr(df_link)
    )
    df_link = df_link.filter(pl.col('link_fdr') <= link_fdr)

    # Calculate PPI FDR
    logger.debug('Calculate PPI FDR')
    ppi_cols = link_cols.copy()
    ppi_cols.remove('cl_pos_p1')
    ppi_cols.remove('cl_pos_p2')
    ppi_merge_cols = [c for c in df_link.columns if c not in ppi_cols+never_agg_cols]
    df_ppi = df_link.with_columns(
        protein_p1=pl.col('protein_p1').list.unique().list.sort(),
        protein_p2=pl.col('protein_p2').list.unique().list.sort(),
    ).with_columns(
        **{  # Swap proteins again after unique
            c1: pl.when(
                pl.col('protein_p1').list.join(';') > pl.col('protein_p2').list.join(';')
            ).then(c2).otherwise(c1)
            for c1, c2 in zip(
                [c for c in df_link.columns if c.endswith('_p1')] +  # Swap _p1 with _p2
                [c for c in df_link.columns if c.endswith('_p2')],   # Swap _p2 with _p1s
                [c for c in df_link.columns if c.endswith('_p2')] +  # Swap _p1 with _p2
                [c for c in df_link.columns if c.endswith('_p1')],   # Swap _p2 with _p1
            )
        },
    ).group_by(ppi_cols).agg(
        *first_aggs,
        *[
            pl.col(c).flatten()
            for c in ppi_merge_cols
        ],
        score=aggs['ppi']
    )
    df_ppi = df_ppi.with_columns(
        ppi_fdr = single_grouped_fdr(df_ppi)
    )
    df_ppi = df_ppi.filter(pl.col('ppi_fdr') <= ppi_fdr)

    # Back-fitler levels
    if filter_back:
        df_link = df_link.join(
            df_ppi.select(ppi_cols),
            on=ppi_cols,
            how='left'
        )
        df_pep = df_pep.join(
            df_link.select(link_cols),
            on=link_cols,
            how='left'
        )
        df_csm = df_csm.join(
            df_pep.select(pep_cols),
            on=pep_cols,
            how='left'
        )

    return {
        'csm': df_csm,
        'pep': df_pep,
        'prot': df_prot,
        'link': df_link,
        'ppi': df_ppi,
    }


def single_grouped_fdr(df: pl.DataFrame | pd.DataFrame, fdr_group_col: str = "fdr_group") -> pl.Series:
    """
    Computes the false discovery rate (FDR) for a given DF.

    Parameters
    ----------
    df : pl.DataFrame | pd.DataFrame
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
        class_df = class_df.with_columns(
            single_fdr(class_df)
        )
        fdr_with_order = fdr_with_order.extend(class_df)

    return fdr_with_order.sort(order_col)['fdr']


def single_fdr(df: pl.DataFrame | pd.DataFrame) -> pl.Series:
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
        (working_df['TD'].cast(pl.Int8).cum_sum() - working_df['DD'].cast(pl.Int8).cum_sum())
        / working_df['TT'].cast(pl.Int8).cum_sum()
    )
    working_df = working_df.with_columns(
        fdr = fdr_raw.reverse().cum_min().reverse()
    )
    return working_df.sort(order_col)['fdr']
