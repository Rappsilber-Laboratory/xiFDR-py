import pandas as pd
import polars as pl
from polars import col


def full_fdr(df: pl.DataFrame | pd.DataFrame,
             psm_fdr:float = 1.0,
             pep_fdr:float = 1.0,
             prot_fdr:float = 1.0,
             link_fdr:float = 1.0,
             ppi_fdr:float = 1.0,
             decoy_adjunct:str = 'REV_'):
    # Convert non-polars DFs
    if not isinstance(df, pl.DataFrame):
        df: pl.DataFrame = pl.DataFrame(df)

    # Convert semicolon separated string columns to lists
    list_cols = [
        'protein_p1', 'protein_p2',
        'start_pos_p1', 'start_pos_p1',
    ]
    for c in list_cols:
        if not isinstance(df[c].dtype, pl.List):
            df = df.with_columns(
                col(c).cast(pl.String).str.split(';')
            )
        df = df.with_columns(
            col(c).list.sort()
        )

    # Check for required columns
    required_columns = [
        'score',  # Match score
        'decoy_p1', 'decoy_p2',  # Target/decoy classification
        'decoy_class',  # Self- or between-link
        'charge',  # Precursor charge
        'cl_pos_p1', 'cl_pos_p2',  # Positions of crosslinks in the protein sequences
        'start_pos_p1', 'start_pos_p2',  # Position of peptides in proteins origins
        'sequence_p1', 'sequence_p2',  # Peptide sequences including modifications
        'protein_p1', 'protein_p2',  # Protein origins of the peptides
        'coverage_p1', 'coverage_p2'  # Fragment coverage for the peptides
    ]

    # Check for required columns
    missing_columns = [
        c for c in required_columns
        if c not in df.columns
    ]
    if len(missing_columns) > 0:
        raise Exception(f'Missing required columns: {missing_columns}')

    # Calculate one-hot encoded target/decoy labels
    df.with_columns(
        ((~col('decoy_p1')) & (~col('decoy_p2'))).alias('TT'),
        (col('decoy_p1') & col('decoy_p2')).alias('DD')
    )
    df.with_columns(
        ((~col('DD')) & (~col('TT'))).alias('TD')
    )

    # Aggregate unique PSMs
    never_agg_cols = ['fdr_group', 'decoy_class', 'TT', 'TD', 'DD']
    first_aggs = [
        col(c).get(0)
        for c in never_agg_cols
    ]
    never_agg_cols += ['score']

    psm_cols = required_columns.copy()
    psm_cols.remove('score')
    psm_cols.remove('decoy_class')
    df_psm = df.sort('score', descending=True).unique(subset=psm_cols, keep='first')

    # Calculate PSM FDR and cutoff
    print('Calculate PSM FDR and cutoff')
    df_psm = df_psm.with_columns(
        single_bi_fdr(df_psm).alias('psm_fdr')
    )
    df_psm = df_psm.filter(col('psm_fdr') <= psm_fdr)

    # Calculate peptide FDR and filter
    print('Calculate peptide FDR and filter')
    pep_cols = psm_cols.copy()
    pep_cols.remove('charge')
    pep_merge_cols = [c for c in df_psm.columns if c not in pep_cols+never_agg_cols]
    df_pep = df_psm.group_by(pep_cols).agg(
        col('score').max(),
        *first_aggs,
        *[
            col(c).flatten()
            for c in pep_merge_cols
        ]
    )
    df_pep = df_pep.with_columns(
        single_bi_fdr(df_pep).alias('pep_fdr')
    )
    df_pep = df_pep.filter(col('pep_fdr') <= pep_fdr)

    # Unpack filtered peptide level to back-filtered PSM level
    df_psm_filtered = df_pep

    # Distribute score by peptide coverage for protein group FDR
    df_psm_filtered = df_psm_filtered.with_columns(
        (
            df_psm_filtered['coverage_p1'] / (
                df_psm_filtered['coverage_p1'] + df_psm_filtered['coverage_p2']
            )
        ).alias('coverage_p1_prop')
    )
    df_psm_filtered = df_psm_filtered.with_columns(
        (df_psm_filtered['score'] * df_psm_filtered['coverage_p1_prop']).alias('protein_score_p1'),
        (df_psm_filtered['score'] / df_psm_filtered['coverage_p1_prop']).alias('protein_score_p2')
    )

    # Construct protein (group) DF
    df_prot_p1 = df_psm_filtered.select([
        'protein_p1', 'protein_score_p1', 'decoy_p1'
    ]).rename({
        'protein_p1': 'protein',
        'protein_score_p1': 'score',
        'decoy_p1': 'decoy',
    })

    df_prot_p2 = df_psm_filtered.select([
        'protein_p2', 'protein_score_p2', 'decoy_p2'
    ]).rename({
        'protein_p2': 'protein',
        'protein_score_p2': 'score',
        'decoy_p2': 'decoy',
    })

    ## Calculate and filter protein group FDR as linear FDR
    print('Calculate and filter protein group FDR as linear FDR')
    df_prot = pl.concat([
        df_prot_p1,
        df_prot_p2
    ])
    df_prot = df_prot.group_by(['protein', 'decoy']).agg(
        (col('score')**2).sum().sqrt()
    )
    df_prot = df_prot.with_columns(
        col('decoy').alias('DD'),
        col('decoy').not_().alias('TT'),
        pl.lit(False).alias('TD')  # Abuse CL-FDR for linear case
    )
    df_prot = df_prot.with_columns(
        single_fdr(df_prot).alias('prot_fdr')
    )
    df_prot = df_prot.filter(col('prot_fdr') <= prot_fdr)

    passed_prots = df_prot['protein']
    passed_prots = passed_prots.list.join(';')
    passed_prots = passed_prots.str.replace_all(decoy_adjunct, '')
    passed_prots = passed_prots.str.split(';')
    passed_prots = passed_prots.list.sort()
    passed_prots = passed_prots.list.join(';')
    passed_prots = passed_prots.unique().alias('passed_prots')

    ## Filter left over peptide pairs (using polar)
    print('Filter left over peptide pairs (using polar)')
    df_pep = df_pep.with_columns(
        (
            col('protein_p1')
                # Replace decoy_adjunct
                .list.join(';')
                .str.replace_all(decoy_adjunct, '')
                # Sort base protein names
                .str.split(';')
                .list.sort()
                # Join to protein group
                .list.join(';')
        ).alias('base_protein_p1'),
        (
            col('protein_p2')
                # Replace decoy_adjunct
                .list.join(';')
                .str.replace_all(decoy_adjunct, '')
                # Sort base protein names
                .str.split(';')
                .list.sort()
                # Join to protein group
                .list.join(';')
        ).alias('base_protein_p2'),
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

    # Disregard linear peptides from here on
    df_pep = df_pep.filter(
        col('base_protein_p2').is_not_null()
    )

    # Calculate link FDR and cutoff
    print('Calculate link FDR and cutoff')
    link_cols = pep_cols.copy()
    #link_cols.remove('aa_len_p1')
    #link_cols.remove('aa_len_p2')
    link_cols.remove('start_pos_p1')
    link_cols.remove('start_pos_p2')
    link_merge_cols = [c for c in df_pep.columns if c not in link_cols+never_agg_cols]
    df_link = df_pep.group_by(link_cols).agg(
        (col('score')**2).sum().sqrt(),
        *first_aggs,
        *[
            col(c).flatten()
            for c in link_merge_cols
        ]
    )
    df_link = df_link.with_columns(
        single_bi_fdr(df_link).alias('link_fdr')
    )
    df_link = df_link.filter(col('link_fdr') <= link_fdr)

    # Calculate PPI FDR
    print('Calculate PPI FDR')
    ppi_cols = link_cols.copy()
    ppi_cols.remove('cl_pos_p1')
    ppi_cols.remove('cl_pos_p2')
    ppi_merge_cols = [c for c in df_link.columns if c not in ppi_cols+never_agg_cols]
    df_ppi = df_link.group_by(ppi_cols).agg(
        (col('score')**2).sum().sqrt(),
        *first_aggs,
        *[
            col(c).flatten()
            for c in ppi_merge_cols
        ]
    )
    df_ppi = df_ppi.with_columns(
        single_bi_fdr(df_ppi).alias('ppi_fdr')
    )
    df_ppi = df_ppi.filter(col('ppi_fdr') <= ppi_fdr)

    return df_psm, df_pep, df_link, df_ppi, passed_prots


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
    fdr_with_order = fdr_with_order.with_columns(pl.lit(0.0).alias('fdr'))
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
    fdr_raw = (working_df['TD'].cum_sum() - working_df['DD'].cum_sum()) / working_df['TT'].cum_sum()
    working_df = working_df.with_columns(
        fdr_raw.reverse().cum_min().reverse().alias('fdr')
    )
    return working_df.sort(order_col)['fdr']
