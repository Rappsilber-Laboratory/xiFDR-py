import os
from functools import partial
import logging
import psutil

import numpy as np
from polars import col
import polars as pl
from contextlib import closing
from multiprocessing import get_context
from .fdr import full_fdr
from .utils.column_preparation import prepare_columns
from .utils.knee_finder import find_knees

logger = logging.getLogger(__name__)

def boost(df: pl.DataFrame,
          csm_fdr: (float, float) = (0.0, 1.0),
          pep_fdr: (float, float) = (0.0, 1.0),
          prot_fdr: (float, float) = (0.0, 1.0),
          link_fdr: (float, float) = (0.0, 1.0),
          ppi_fdr: (float, float) = (0.0, 1.0),
          boost_cols: list = [],
          neg_boost_cols: list = [],
          boost_level: str = "ppi",
          boost_between: bool = True,
          method: str = "manhattan",
          countdown: int = 3,
          points: int = 10,
          n_jobs: int = -1,
          **kwargs) -> (float, float, float, float, float):
    """
    Find the best FDR cutoffs to optimize results for a certain FDR level.

    Parameters
    ----------
    df
        CSM DataFrame
    csm_fdr
        Search range for CSM FDR level cutoff
    pep_fdr
        Search range for peptide FDR level cutoff
    prot_fdr
        Search range for protein FDR level cutoff
    link_fdr
        Search range for residue link FDR level cutoff
    ppi_fdr
        Search range for protein pair FDR level cutoff
    boost_level
        FDR level tp boost for
    boost_between
        Whether to boost for between links
    method
        Search algorithm to use
    countdown
        Number interation without improvement to stop
    points
        Number of FDR cutoffs to search in one iteration
    n_jobs
        Number of threads to use

    Returns
    -------
        Returns a tuple with the optimal FDR levels.
    """
    if method == 'manhattan':
        return boost_manhattan(
            df=df,
            csm_fdr=csm_fdr,
            pep_fdr=pep_fdr,
            prot_fdr=prot_fdr,
            link_fdr=link_fdr,
            ppi_fdr=ppi_fdr,
            boost_cols=boost_cols,
            neg_boost_cols=neg_boost_cols,
            boost_level=boost_level,
            boost_between=boost_between,
            countdown=countdown,
            points=points,
            n_jobs=n_jobs,
            **kwargs
        )
    else:
        raise ValueError(f'Unkown boosting method: {method}')

def boost_manhattan(df: pl.DataFrame,
                    csm_fdr: (float, float) = (0.0, 1.0),
                    pep_fdr: (float, float) = (0.0, 1.0),
                    prot_fdr: (float, float) = (0.0, 1.0),
                    link_fdr: (float, float) = (0.0, 1.0),
                    ppi_fdr: (float, float) = (0.0, 1.0),
                    boost_cols: list = [],
                    neg_boost_cols: list = [],
                    boost_level: str = "ppi",
                    boost_between: bool = True,
                    countdown: int = 3,
                    points: int = 10,
                    n_jobs: int = -1,
                    **kwargs):
    df = prepare_columns(df)
    param_ranges = (
        csm_fdr,
        pep_fdr,
        prot_fdr,
        link_fdr,
        ppi_fdr
    )
    param_ranges += tuple((0.0, 1.0) for _ in boost_cols)
    param_ranges += tuple((0.0, 1.0) for _ in neg_boost_cols)

    # Init best_params
    best_params = []
    for mini, maxi in param_ranges[:5]:
        best_params += [maxi]

    # Figure out knee points for starting
    df = prepare_columns(df)
    knee_points = find_knees(df.filter(pl.col ('fdr_group') == 'between'), **kwargs)
    for i, p in enumerate(knee_points):
        best_params[i] = p
        # Clip to param max
        best_params[i] = min(
            param_ranges[i][1],
            best_params[i],
        )
        # Clip to param min
        best_params[i] = max(
            param_ranges[i][0],
            best_params[i],
        )

    # Init boost col levels
    for _ in boost_cols:
        best_params += [0]
    for _ in neg_boost_cols:
        best_params += [1]

    # Init spreads
    search_spreads = np.array([])
    for r_from, r_to in param_ranges:
        r_spread = (r_to-r_from)/2
        search_spreads = np.append(search_spreads, [r_spread])

    opt_wrapper = partial(
        _optimization_template,
        df=df,
        boost_cols=boost_cols,
        neg_boost_cols=neg_boost_cols,
        boost_level=boost_level,
        boost_between=boost_between,
        **kwargs
    )

    n_params = len(param_ranges)
    best_result = opt_wrapper(best_params)
    logger.info(f'Initial result ({best_result}) for params: {best_params}')
    current_countdown = countdown
    if n_jobs <= 0:
        max_mem_cpu = int(psutil.virtual_memory().available // df.estimated_size())
        n_jobs = max(max_mem_cpu, 1)
        n_jobs = min(n_jobs, os.cpu_count())
        logger.info(f"Using {n_jobs} CPUs based on available memory.")
    with closing(get_context('spawn').Pool(n_jobs)) as pool:
        while True:
            grids = []
            for param_index in range(n_params):
                # Generate grid for parameter
                grid = [
                    [x] * points for x in best_params
                ]
                param_from = max(
                    param_ranges[param_index][0],
                    best_params[param_index] - search_spreads[param_index]
                )
                param_to = min(
                    param_ranges[param_index][1],
                    best_params[param_index] + search_spreads[param_index]
                )
                grid[param_index] = np.unique(
                    np.linspace(param_from, param_to, points)
                ).tolist()

                n_unique = len(grid[param_index])

                grid = [
                    x[:n_unique] for x in grid
                ]

                # Transpose grid to correct format and append
                grids.append(
                    np.transpose(grid)
                )
            grid_flat = np.vstack(
                grids
            )
            # Run FDR calculation (possibly in parallel)
            results_flat = list(pool.map(opt_wrapper, grid_flat))
            # Copy last best parameters to generate new best
            top_params = best_params.copy()
            grid_top_results = []
            # Iterate over results for certain parameters
            for grid_i, grid in enumerate(grids):
                # Get range in flat results
                from_index = sum([
                    len(g)
                    for g in grids[:grid_i]
                ])
                to_index = from_index + len(grid)
                # Cut out according results
                grid_results = results_flat[from_index:to_index]
                # Get index of best result for this parameter
                grid_top_index = np.argmin(grid_results)
                # Get best params config and result for this parameter
                grid_top_params = grid[grid_top_index]
                grid_top_result = grid_results[grid_top_index]
                grid_top_results.append(grid_top_result)
                # Update next best parameters if result improved
                if best_result is None or grid_top_result < best_result:
                    top_params[grid_i] = grid_top_params[grid_i]
            top_result = min(grid_top_results)
            if best_result is None or top_result < best_result:
                param_diff = np.abs(np.array(best_params)-np.array(top_params))
                search_spreads[param_diff!=0] = param_diff[param_diff!=0] * 2
                best_result = top_result
                best_params = top_params
                current_countdown = countdown
                logger.info(f'Better score found ({best_result}) for params: {best_params}')
            else:
                if current_countdown == 0:
                    break
                search_spreads[search_spreads<0.05] *= countdown-current_countdown+2
                search_spreads[search_spreads>=0.05] *= .8
                current_countdown -= 1
                logger.info(f'No improvement for iteration. Countdown: {current_countdown}')

    return best_params


def _optimization_template(cutoffs,
                           df: pl.DataFrame,
                           min_len: int = 5,
                           unique_csm: bool = True,
                           boost_cols: list = [],
                           neg_boost_cols: list = [],
                           boost_level: str = "ppi",
                           boost_between: bool = True,
                           td_prob: int = 2,
                           td_prot_prob: int = 10,
                           td_dd_ratio: float = 1.0) -> float:
    fdrs = cutoffs[:5]
    col_levels = cutoffs[5:]
    neg_col_levels = col_levels[len(boost_cols):]

    for i, c in enumerate(boost_cols):
        df = df.filter(
            (
                    (pl.col(c)-pl.col(c).min()) /
                    (pl.col(c).max()-pl.col(c).min())
            ) >= col_levels[i]
        )

    for i, c in enumerate(neg_boost_cols):
        df = df.filter(
            (
                    (pl.col(c)-pl.col(c).min()) /
                    (pl.col(c).max()-pl.col(c).min())
            ) <= neg_col_levels[i]
        )
    result_all = full_fdr(
        df, *fdrs,
        min_len=min_len,
        unique_csm=unique_csm,
        prepare_column=False,
        td_prob=0,
        td_prot_prob=td_prot_prob,
        td_dd_ratio=0
    )
    result = result_all[boost_level]
    if boost_between:
        result = result.filter(col('fdr_group') == 'between')
    tt = len(result.filter(col('TT')))
    td = len(result.filter(col('TD')))
    dd = len(result.filter(col('DD')))
    tp = tt - td + dd
    logger.debug(
        f'Estimated true positive matches: {tp}\n'
        f'Parameters: {cutoffs}'
    )

    # Check for probabilities. If the result does not match the TD/DD probabilities
    # we return the result divided by the df size to indicate that this is better than
    # nothing but should be dropped as soon as a proper result is found.
    for li, l in [(0, 'csm'), (1, 'pep'), (3, 'link'), (4, 'ppi')]:
        for g in ['self', 'between']:
            gl_df = result_all[l].filter(pl.col('fdr_group')==g)
            gl_tt = gl_df.filter('TT').height
            gl_td = gl_df.filter('TD').height
            gl_dd = gl_df.filter('DD').height
            td_prob_bad = gl_tt*cutoffs[li] < td_prob
            dd_prob_bad = gl_dd*td_dd_ratio > gl_td
            if td_prob_bad or dd_prob_bad:
                return -tp/df.height

    return -tp