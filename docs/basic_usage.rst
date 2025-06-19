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

TODO

Running a boosted multi-level FDR
---------------------------------

TODO
