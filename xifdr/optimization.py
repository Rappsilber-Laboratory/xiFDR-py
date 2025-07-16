import logging
import numpy as np
from scipy._lib._util import MapWrapper

logger = logging.getLogger(__name__)

def manhattan(func,
              ranges: tuple[tuple[float]],
              args=(),
              kwargs={},
              points:int = 5,
              workers:int = 1,
              countdown:int = 3):
    param_index:int = 0
    n_params:int = len(ranges)
    best_params = []
    best_result = None
    search_spreads = []
    # Initialize search_spreads and best_params
    for r_from, r_to in ranges:
        r_spread = (r_to-r_from)/2
        r_mid = r_from+r_spread
        best_params += [r_mid]
        search_spreads += [r_spread]

    func_wrapped = ManhattanWrapper(func, args, kwargs)

    improved = False
    current_countdown = countdown
    with MapWrapper(pool=workers) as mapper:
        while True:
            grid = [
                [x]*points for x in best_params
            ]
            param_from = max(
                ranges[param_index][0],
                best_params[param_index] - search_spreads[param_index]
            )
            param_to = min(
                ranges[param_index][1],
                best_params[param_index] + search_spreads[param_index]
            )
            grid[param_index] = np.unique(np.linspace(param_from, param_to, points)).tolist()
            n_unique = len(grid[param_index])
            grid = [
                x[:n_unique] for x in grid
            ]
            grid = np.transpose(grid)
            # Run parameters
            results = list(mapper(func_wrapped, grid))
            top_index = np.argmin(results)
            top_run = results[top_index]
            top_params = grid[top_index]
            if best_result is None or top_run < best_result:
                best_result = top_run
                best_params[param_index] = top_params[param_index]
                improved = True
                logger.info(f'Better score found ({best_result}) for params: {best_params}')
            search_spreads[param_index] *= .8
            param_index += 1
            if param_index >= n_params:
                if not improved:
                    current_countdown -= 1
                    logger.info(f'No improvement for iteration. Countdown: {current_countdown}')
                    if current_countdown == 0:
                        break
                else:
                    current_countdown = countdown
                param_index=0
                improved = False
    return best_params, best_result


def independent_gird(func,
              ranges: tuple[tuple[float]],
              args=(),
              kwargs={},
              points:int = 5,
              workers:int = 1,
              countdown:int = 3):
    n_params:int = len(ranges)
    best_params = []
    best_result = None
    search_spreads = []
    # Initialize search_spreads and best_params
    for r_from, r_to in ranges:
        r_spread = (r_to-r_from)/2
        r_mid = r_from+r_spread
        best_params += [r_mid]
        search_spreads += [r_spread]

    func_wrapped = ManhattanWrapper(func, args, kwargs)

    current_countdown = countdown
    with MapWrapper(pool=workers) as mapper:
        while True:
            grids = []
            for param_index in range(n_params):
                # Generate grid for parameter
                grid = [
                    [x] * points for x in best_params
                ]
                param_from = max(
                    ranges[param_index][0],
                    best_params[param_index] - search_spreads[param_index]
                )
                param_to = min(
                    ranges[param_index][1],
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
            results_flat = list(mapper(func_wrapped, grid_flat))
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
                to_index = from_index+len(grid)
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
                best_result = top_result
                best_params = top_params
                current_countdown = countdown
                logger.info(f'Better score found ({best_result}) for params: {best_params}')
            else:
                current_countdown -= 1
                logger.info(f'No improvement for iteration. Countdown: {current_countdown}')
                if current_countdown == 0:
                    break
            search_spreads *= np.multiply(search_spreads, .8)
    return best_params, best_result



class ManhattanWrapper:
    def __init__(self, f, args, kwargs):
        self.f = f
        self.args = [] if args is None else args
        self.kwargs = {} if kwargs is None else kwargs

    def __call__(self, x):
        # flatten needed for one dimensional case.
        return self.f(np.asarray(x).flatten(), *self.args, **self.kwargs)
