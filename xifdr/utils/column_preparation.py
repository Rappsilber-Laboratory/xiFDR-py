import polars as pl
from polars import col


def prepare_columns(df):
    """Prepares and processes a Polars DataFrame for protein-protein interaction analysis.

    This function ensures the proper format of protein columns, calculates crosslink positions,
    sorts protein lists, swaps peptides based on predefined criteria, and computes various scores
    and classification labels.

    Parameters
    ----------
    df : pl.DataFrame
        A Polars DataFrame containing protein interaction data. If not already a Polars DataFrame,
        it will be converted.

    Returns
    -------
    pl.DataFrame
        The processed DataFrame with formatted columns, calculated positions, sorted lists,
        swapped peptides, and additional computed fields.

    Notes
    -----
    - Converts semicolon-separated protein columns into lists.
    - Computes crosslink positions by adjusting start positions.
    - Sorts list columns based on protein group and start position order.
    - Swaps peptides based on a custom string comparison mask.
    - Computes one-hot encoded labels (`TT`, `TD`, `DD`) for classification.
    - Ensures a positive score and assigns dummy coverage values if missing.
    - Computes proportional protein scores based on coverage.
    """
    if not isinstance(df, pl.DataFrame):
        df: pl.DataFrame = pl.DataFrame(df)

    # Convert semicolon separated string columns to lists
    list_cols_1 = [
        'protein_p1', 'start_pos_p1'
    ]
    list_cols_2 = [
        'protein_p2', 'start_pos_p2'
    ]
    list_cols = list_cols_1 + list_cols_2
    for c in list_cols:
        if not isinstance(df[c].dtype, pl.List):
            df = df.with_columns(
                col(c).cast(pl.String).str.replace_all(
                    '[ ]*',
                    ''
                ).str.split(';')
            )

    # Generate fdr_group if not present
    if 'fdr_group' not in df.columns:
        df = df.with_columns(
            fdr_group=(
                (pl.col('protein_p1').list.set_intersection(
                    pl.col('protein_p2')
                ).list.len()==0).cast(pl.String).replace(
                    ['true', 'false'],
                    ['between', 'self']
                )
            )
        ).with_columns(
            fdr_group=pl.when(pl.col('protein_p2').eq([]) | pl.col('protein_p2').is_null()).then(
                pl.lit('linear')
            ).otherwise(
                pl.col('fdr_group')
            )
        )

    # Create decoy_class column if not present
    if 'decoy_class' not in df.columns:
        df = df.with_columns(
            decoy_class=pl.when(
                col('decoy_p1') & col('decoy_p2')
            ).then(
                pl.lit('DD')
            ).when(
                col('decoy_p1').not_() & col('decoy_p2').not_()
            ).then(
                pl.lit('TT')
            ).otherwise(
                pl.lit('TD')
            )
        )

    # Sort list columns by protein group order
    start_rank_p1 = pl.col('start_pos_p1').list.eval(pl.element().rank(method="dense"))
    start_rank_p2 = pl.col('start_pos_p2').list.eval(pl.element().rank(method="dense"))
    prot_rank_p1 = pl.col('protein_p1').list.eval(pl.element().rank(method="dense"))
    prot_rank_p2 = pl.col('protein_p2').list.eval(pl.element().rank(method="dense"))
    group_p1_rank = (
        prot_rank_p1 * start_rank_p1.list.max().add(1) + start_rank_p1
    )
    group_p2_rank = (
        prot_rank_p2 * start_rank_p2.list.max().add(1) + start_rank_p2
    )
    df = df.with_columns(
        col('protein_p2').fill_null([]),
        col('start_pos_p2').fill_null([]),
    )

    df = df.with_columns(
        [
            col(c).list.gather(group_p1_rank.list.eval(pl.element().arg_sort()))
            for c in list_cols_1
        ]
    ).with_columns(
        [
            col(c).list.gather(group_p2_rank.list.eval(pl.element().arg_sort()))
            for c in list_cols_2
        ]
    )

    # Calculate crosslink position in protein
    df = df.with_columns(
        cl_pos_p1 = col('start_pos_p1').cast(pl.List(pl.Int64)) + col('link_pos_p1') - 1,
        cl_pos_p2 = col('start_pos_p2').cast(pl.List(pl.Int64)) + col('link_pos_p2') - 1,
    )

    # Swap peptides based on joined protein group
    protein_p1_str = col('protein_p1').list.join(';')
    protein_p2_str = col('protein_p2').list.join(';')
    link_pos_p1_str = col('link_pos_p1').cast(pl.String)
    link_pos_p2_str = col('link_pos_p2').cast(pl.String)
    cl_pos_p1_str = col('cl_pos_p1').cast(pl.List(pl.String)).list.join(';')
    cl_pos_p2_str = col('cl_pos_p2').cast(pl.List(pl.String)).list.join(';')
    # Calculate paddings for joint swap string comparison
    pad_prot = max(
        df.select(protein_p1_str.str.len_chars()).to_series().max(),
        df.select(protein_p2_str.str.len_chars()).to_series().max(),
    )
    pad_link = max(
        df.select(link_pos_p1_str.str.len_chars()).to_series().max(),
        df.select(link_pos_p2_str.str.len_chars()).to_series().max(),
    )
    pad_seq = max(
        df.select(col('sequence_p1').str.len_chars()).to_series().max(),
        df.select(col('sequence_p2').fill_null('').str.len_chars()).to_series().max(),
    )
    pad_cl = max(
        df.select(cl_pos_p1_str.str.len_chars()).to_series().max(),
        df.select(cl_pos_p2_str.str.len_chars()).to_series().max(),
    )
    # Generate swap mask
    swap_mask = df.select(
        (
            protein_p1_str.str.pad_start(pad_prot,' ')+
            cl_pos_p1_str.str.pad_start(pad_cl,' ')+
            link_pos_p1_str.str.pad_start(pad_link,' ')+
            col('sequence_p1').str.pad_start(pad_seq,' ')
        ) > (
            protein_p2_str.str.pad_start(pad_prot,' ')+
            cl_pos_p2_str.str.pad_start(pad_cl,' ')+
            link_pos_p2_str.str.pad_start(pad_link,' ')+
            col('sequence_p2').str.pad_start(pad_seq,' ')
        )
    ).to_series()
    # Swap peptide specific columns
    pair_cols1 = ['sequence_p1', 'protein_p1', 'start_pos_p1', 'link_pos_p1', 'cl_pos_p1', 'decoy_p1']
    pair_cols2 = ['sequence_p2', 'protein_p2', 'start_pos_p2', 'link_pos_p2', 'cl_pos_p2', 'decoy_p2']
    for c1, c2 in zip(pair_cols1, pair_cols2):
        df = df.with_columns(
           pl.when(swap_mask).then(col(c2)).otherwise(col(c1)).alias(c1),
           pl.when(swap_mask).then(col(c1)).otherwise(col(c2)).alias(c2),
        )

    # Calculate one-hot encoded target/decoy labels
    df = df.with_columns(
        TT=(pl.col('decoy_class')=='TT'),
        TD=(pl.col('decoy_class')=='TD'),
        DD=(pl.col('decoy_class')=='DD'),
    )

    # Calculate decoy_class column
    df = df.with_columns(
        decoy_class = pl.when(
            col('decoy_p1') & (col('decoy_p2') | col('decoy_p2').is_null())
        ).then(pl.lit('DD')).when(
            (~col('decoy_p1')) & ((~col('decoy_p2')) | col('decoy_p2').is_null())
        ).then(pl.lit('TT')).otherwise(pl.lit('TD'))
    )

    # Make score positive
    df = df.with_columns(col('score') - col('score').min())

    # Put in dummy coverage if none provided
    if 'coverage_p1' not in df.columns or 'coverage_p2' not in df.columns:
        df = df.with_columns(
            coverage_p1 = pl.lit(0.5),
            coverage_p2 = pl.lit(0.5)
        )

    coverage_p1_prop = col('coverage_p1') / (col('coverage_p1') + col('coverage_p2'))
    coverage_p2_prop = col('coverage_p2') / (col('coverage_p1') + col('coverage_p2'))
    df = df.with_columns(
        protein_score_p1 = col('score') * coverage_p1_prop,
        protein_score_p2 = col('score') * coverage_p2_prop
    )

    return df