Basic usage
===========

Let's quickly demonstrate how to use this package.

Installation
------------

Use `pip` to install xiFDR from PyPi:

.. code-block:: shell

    $ pip install xifdr

Input format
------------

The full FDR calculation and the boosting methods both support polars or pandas DataFrames. The following column names
play special roles in the FDR calculcation:

.. table::
    :widths: auto

    ============  =========  ====================================================
    column name   required?  function
    ============  =========  ====================================================
    score         yes        score for FDR calculation
    sequence_p1   yes        sequence for peptide 1 including modifications
    sequence_p2   yes        sequence for peptide 2 including modifications
    start_pos_p1  yes        position(s) of peptide 1 in the according protein(s)
    start_pos_p2  yes        position(s) of peptide 2 in the according protein(s)
    link_pos_p1   yes        link position in the sequence of peptide 1
    link_pos_p2   yes        link position in the sequence of peptide 2
    charge        yes        precursor charge
    protein_p1    yes        protein(s) related to peptide 1
    protein_p2    yes        protein(s) related to peptide 2
    decoy_p1      yes        decoy indicator for peptide 1
    decoy_p2      yes        decoy indicator for peptide 2
    fdr_group     no         groups for crosslink FDR calculation
    coverage_p1   no         fragment coverage of peptide 1 (default: 0.5)
    coverage_p2   no         fragment coverage of peptide 2 (default: 0.5)
    ============  =========  ====================================================


Running a full multi-level FDR
------------------------------

To run a standard multi-level FDR calculation with static cutoffs, use the ``full_fdr`` function. 

.. code-block:: python

    import polars as pl
    from xifdr.fdr import full_fdr
    
    # Load your dataframe
    df = pl.read_parquet("my_csm_data.parquet")
    
    # Calculate FDR at all levels with specific cutoffs
    results = full_fdr(
        df,
        csm_fdr=0.05,
        pep_fdr=0.05,
        prot_fdr=0.01,
        link_fdr=0.05,
        ppi_fdr=0.05
    )
    
    # The result is a dictionary of DataFrames for each level
    print(results['csm'].head())
    print(results['prot'].head())

Running a boosted multi-level FDR
---------------------------------

If you want xiFDR to automatically find the optimal set of cutoffs that maximizes the number of true positives at a given FDR level (e.g., protein-pair level), use the ``boost`` function. 

.. code-block:: python

    from xifdr.boosting import boost
    
    # Find the best cutoffs for a 5% PPI FDR, searching within the given ranges
    cutoffs = boost(
        df,
        csm_fdr=(0.0, 0.2),
        pep_fdr=(0.0, 0.2),
        link_fdr=(0.05, 0.05),
        ppi_fdr=(0.05, 0.05),
        boost_cols=['coverage_p1', 'coverage_p2'],
        neg_boost_cols=['charge'],
        points=10,
        n_jobs=-1
    )
    print("Optimal cutoffs:", cutoffs)
    
If you want to perform this optimization separately for ``self`` and ``between`` interactions, you can use ``group_boost`` along with ``group_full_fdr``:

.. code-block:: python

    from xifdr.boosting import group_boost
    from xifdr.fdr import group_full_fdr
    
    # Optimize separately for self and between
    params_dict = group_boost(
        df,
        csm_fdr=(0.0, 0.2),
        ppi_fdr=(0.05, 0.05),
        boost_cols=['coverage_p1', 'coverage_p2']
    )
    
    # Apply both sets of parameters and merge the final DataFrames
    results = group_full_fdr(
        df,
        cutoffs_self=params_dict['self'],
        cutoffs_between=params_dict['between'],
        boost_cols=['coverage_p1', 'coverage_p2']
    )
