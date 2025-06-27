How it works
============

To get from CSMs to residue links or PPIs xiFDR needs to aggregate CSMs. This is done in multiple stages/levels
where on each level an FDR cutoff may be applied. Below you find a summary for each aggregation level.

Aggregation levels
------------------

CSM level
"""""""""

The CSM level is the plain input data to the FDR calculation. A lower FDR cutoff here can greatly reduce
computation time.

Peptide level
"""""""""""""

To aggregate from the CSM level to the peptide level xiFDR combines all CSMs with the same peptide sequences (including
modifications), crosslink positions inside the sequence, positions of the sequences in the according proteins
**diregarding** the scores and charges of the individual matches.

Protein level
"""""""""""""

This is special in the sense, that it does not operate on crosslinks but on the proteins involved in the peptide level
crosslinks. This is done by splitting each peptide level match into the linked proteins and distributing the peptide
level score based on the fragment coverage for the according peptide level link. If no fragment coverage is supplied
in the input DataFrame, the score is divided 50/50 for the proteins. After this, same proteins get aggregated and a
linear FDR calculation and filtering is applied. Finally the peptide level matches gets filtered to only include matches
where each interacting protein also passes the protein level FDR cutoff.

.. note::

    For each protein passing the protein FDR filter, the complementing decoy protein must pass as well. If not, the
    decoy proteins would no longer represent the false positives within the targets.

Link level
""""""""""

This level aggregated the protein FDR filtered matches to links of specific position of the complete protein sequence.
For this, aggregation is performed for matches with the same interacting proteins and the same link position in the
protein sequence (i.e. the same *residue*). This level is relevant if you are not only interested in which proteins
interact with each other but also how this interaction looks in 3D.

PPI level
"""""""""

The final aggregation is the protein-protein interaction level. Here, all link level matches with the same interacting
proteins are aggregated.

Aggregation functions
---------------------

By default the scores in each aggregation are performed using the geometric norm:

.. math::

   score_{agg} = \sqrt{\sum (score_i)^2}

However, you can define custom aggregation functions. Please refer to the `API documentation <api.html>`_.

FDR groups
----------

Due to the different nature of self-links (i.e. homomeric [#selfbetween]_)
and between-links (i.e. heterometric [#selfbetween]_) the FDR is calculated separately for these groups.

For similar reasons we also split the protein level FDR in groups of proteins with only between-link, proteins with only
linear or self-link matches and proteins with self-link or linear matches additional to between-links.

.. [#selfbetween] Homomeric links sometimes refer to links explicitly within a single molecule of a protein.
    As we usually don't know whether the link is within one molecule or between multiple molecule of the same protein,
    we rather use the nomenclature of self- and between-links.

Validity checks
---------------

For some datasets, very low FDR cutoffs may result in too little data coming through. This may make confidnent FDR
estimation impossible. To avoid this, we check if the number of target-target matches results in a minimum number of
decoys ``td_prob`` to be observed (statistically).