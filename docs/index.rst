.. xiFDR documentation master file, created by
   sphinx-quickstart on Tue Feb 25 20:23:05 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

xiFDR documentation
===================

.. currentmodule:: xifdr

Version: |release|

xiFDR is a tool for crosslink mass spectrometry (CLMS) false discovery rate (FDR) estimation
and crosslink spectrum match (CSM) aggregation. Usual applications are the extraction of residue links
or protein-protein interactions (PPIs) above a desired significance threshold.
Further, xiFDR supports *boosting* of FDR levels to obtain the largest possible number of residue-links
or PPIs by finding the optimal FDR thresholds.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   basic_usage
   how_it_works
   api
