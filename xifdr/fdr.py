import typing
import logging
import pandas as pd
import polars as pl
import cython
from polars import col
from xifdr.utils.column_preparation import prepare_columns


logger = logging.getLogger(__name__)

def full_fdr(df: pl.DataFrame | pd.DataFrame,
             psm_fdr:float = 1.0,
             pep_fdr:float = 1.0,
             prot_fdr:float = 1.0,
             link_fdr:float = 1.0,
             ppi_fdr:float = 1.0,
             decoy_adjunct:str = 'REV_',
             unique_psm: bool = True,
             filter_back:bool = True,
             prepare_column:bool = True):
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

    # Aggregate unique PSMs
    never_agg_cols = ['fdr_group', 'decoy_class', 'TT', 'TD', 'DD']
    first_aggs = [
        col(c).get(0)
        for c in never_agg_cols
    ]
    never_agg_cols += ['score', 'protein_score_p1', 'protein_score_p2']

    psm_cols = required_columns.copy()
    psm_cols.remove('score')
    psm_cols.remove('start_pos_p1')
    psm_cols.remove('start_pos_p2')
    psm_cols.remove('link_pos_p1')
    psm_cols.remove('link_pos_p2')
    psm_cols += ['cl_pos_p1', 'cl_pos_p2']

    if unique_psm:
        df_psm = df.sort('score', descending=True).unique(subset=psm_cols, keep='first')
    else:
        df_psm = df

    # Calculate PSM FDR and cutoff
    logger.debug('Calculate PSM FDR and cutoff')
    df_psm = df_psm.with_columns(
        psm_fdr = single_bi_fdr(df_psm)
    )
    df_psm = df_psm.filter(col('psm_fdr') <= psm_fdr)

    # Calculate peptide FDR and filter
    logger.debug('Calculate peptide FDR and filter')
    pep_cols = psm_cols.copy()
    pep_cols.remove('charge')
    pep_merge_cols = [c for c in df_psm.columns if c not in pep_cols+never_agg_cols]
    df_pep = df_psm.group_by(pep_cols).agg(
        (col('score')**2).sum().sqrt(),
        (col('protein_score_p1')**2).sum().sqrt(),
        (col('protein_score_p2')**2).sum().sqrt(),
        *first_aggs,
        *[
            col(c).flatten()
            for c in pep_merge_cols
        ]
    )
    df_pep = df_pep.with_columns(
        pep_fdr = single_bi_fdr(df_pep)
    )
    df_pep = df_pep.filter(col('pep_fdr') <= pep_fdr)

    # Construct protein (group) DF
    df_prot_p1 = df_pep.select([
        'protein_p1', 'protein_score_p1', 'decoy_p1'
    ]).rename({
        'protein_p1': 'protein',
        'protein_score_p1': 'score',
        'decoy_p1': 'decoy',
    })

    df_prot_p2 = df_pep.select([
        'protein_p2', 'protein_score_p2', 'decoy_p2'
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
        protein_group = col('protein').list.unique()
    )
    df_prot = df_prot.group_by(['protein_group', 'decoy']).agg(
        (col('score')**2).sum().sqrt(),
        col('protein')
    )
    df_prot = df_prot.with_columns(
        DD = col('decoy'),
        TT = ~col('decoy'),
        TD = pl.lit(False)  # Abuse CL-FDR for linear case
    )
    df_prot = df_prot.with_columns(
        prot_fdr = single_fdr(df_prot)
    )
    df_prot = df_prot.filter(col('prot_fdr') <= prot_fdr)

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
            col('protein_p1')
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
            col('protein_p2')
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

    df_pep.join(
        passed_prots.to_frame(),
        left_on=['base_protein_p1'],
        right_on=['passed_prots'],
        how='left',
        suffix='p1'
    ).join(
        passed_prots.to_frame(),
        left_on=['base_protein_p2'],
        right_on=['passed_prots'],
        how='left',
        suffix='p2'
    )

    # Calculate link FDR and cutoff
    logger.debug('Calculate link FDR and cutoff')
    link_cols = pep_cols.copy()
    link_cols.remove('sequence_p1')
    link_cols.remove('sequence_p2')
    link_merge_cols = [c for c in df_pep.columns if c not in link_cols+never_agg_cols]
    df_link = df_pep.filter(
        col('fdr_group') != "linear" # Disregard linear peptides from here on
    ).group_by(link_cols).agg(
        (col('score')**2).sum().sqrt(),
        *first_aggs,
        *[
            col(c).flatten()
            for c in link_merge_cols
        ]
    )
    df_link = df_link.with_columns(
        link_fdr = single_bi_fdr(df_link)
    )
    df_link = df_link.filter(col('link_fdr') <= link_fdr)

    # Calculate PPI FDR
    logger.debug('Calculate PPI FDR')
    ppi_cols = link_cols.copy()
    ppi_cols.remove('cl_pos_p1')
    ppi_cols.remove('cl_pos_p2')
    ppi_merge_cols = [c for c in df_link.columns if c not in ppi_cols+never_agg_cols]
    df_ppi = df_link.with_columns(
        protein_p1=col('protein_p1').list.unique().list.sort(),
        protein_p2=col('protein_p2').list.unique().list.sort(),
    ).with_columns(
        **{  # Swap proteins again after unique
            c1: pl.when(
                col('protein_p1').list.join(';') > col('protein_p2').list.join(';')
            ).then(c2).otherwise(c1)
            for c1, c2 in zip(
                [c for c in df_link.columns if c.endswith('_p1')] +  # Swap _p1 with _p2
                [c for c in df_link.columns if c.endswith('_p2')],   # Swap _p2 with _p1s
                [c for c in df_link.columns if c.endswith('_p2')] +  # Swap _p1 with _p2
                [c for c in df_link.columns if c.endswith('_p1')],   # Swap _p2 with _p1
            )
        },
    ).group_by(ppi_cols).agg(
        (col('score')**2).sum().sqrt(),
        *first_aggs,
        *[
            col(c).flatten()
            for c in ppi_merge_cols
        ]
    )
    df_ppi = df_ppi.with_columns(
        ppi_fdr = single_bi_fdr(df_ppi)
    )
    df_ppi = df_ppi.filter(col('ppi_fdr') <= ppi_fdr)

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
        df_psm = df_psm.join(
            df_pep.select(pep_cols),
            on=pep_cols,
            how='left'
        )

    return {
        'psm': df_psm,
        'pep': df_pep,
        'prot': df_prot,
        'link': df_link,
        'ppi': df_ppi,
    }


def single_bi_fdr(df: pl.DataFrame | pd.DataFrame) -> pl.Series:
    """
    Computes the false discovery rate (FDR) for a given DF.

    Parameters
    ----------
    df : pl.DataFrame | pd.DataFrame
        The input DF containing columns for TT, TD, DD, decoy_class and score.

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
    for dclass in ['self', 'between', 'linear']:
        class_df = df.filter(
            col('fdr_group') == dclass
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

@cython.cfunc
@cython.boundscheck(False)
@cython.wraparound(False)
def _str_zip(protein_list: typing.List[str],
                 pos_list: typing.List[str]) -> typing.List[str]:
    return [f'{a};{b}' for a,b in zip(protein_list, pos_list)]