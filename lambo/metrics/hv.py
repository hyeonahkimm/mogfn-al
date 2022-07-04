import numpy as np
from pymoo.factory import get_performance_indicator
from botorch.utils.multi_objective import pareto, infer_reference_point


def get_hv_metrics(ref_point, solutions, maximize=True):
    pareto_mask = pareto.is_non_dominated(torch.tensor(solutions) if maximize else -torch.tensor(solutions))
    pareto_front = solutions[pareto_mask]

    if ref_point is None:
        ref_point = infer_reference_point(torch.tensor(pareto_front))
    hv_indicator = get_performance_indicator('hv', ref_point=ref_point)

    new_hypervol = hv_indicator.do(-pareto_targets if maximize else pareto_front)
    
    return pareto_front, new_hypervol
