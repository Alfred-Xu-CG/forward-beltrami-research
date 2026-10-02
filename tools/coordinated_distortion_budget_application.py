"""One incumbent-derived ARAP-budget continuation; unchanged image/point evidence.

This is an instance optimizer package, not a learned encoder or AD through the
optimization trajectory. Its cap and the original frozen evidence are never
selected from anatomical labels. The original optimizer remains unchanged.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F

from tools.coordinated_data_metric_application import _build_evidence, _validate_configuration, _plain
from tools.coordinated_real_case import Evidence
from tools.coordinated_registration_start import load_incumbent
from tools.digital_q1_dhr_distill import identity_vertices
from qcopt.neural_bijection.dense.digital_q1 import q1_corner_determinants
from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map


class BudgetEvidence:
    """Assemble non-ARAP terms directly, retaining differentiable actual ARAP."""
    def __init__(self, evidence):
        self.evidence = evidence

    def __call__(self, vertices):
        original_total, parts = self.evidence(vertices)
        value = self.evidence.image_weight * parts['image']
        value = value + self.evidence.oob_weight * parts['oob']
        if self.evidence.shape_weight:
            value = value + self.evidence.shape_weight * parts['shape']
        if self.evidence.match_weight:
            value = value + self.evidence.match_weight * parts['match']
        return value, dict(parts, original_total=original_total)


def build_pyramid(args, matrix, offset, device):
    full, preprocessing, match_metadata = _build_evidence(args, matrix, offset, device)
    preprocessing['moving_descriptor_prewarp_scope'] = 'once per original area-reduced raw raster scale; never through incumbent'
    evidences = {args.image_side: full}
    for side in sorted(set(args.image_levels)):
        if side == args.image_side:
            continue
        reduce = lambda image: F.interpolate(image, size=(side, side), mode='area')
        item = Evidence(reduce(full.fixed), reduce(full.moving), full.matrix, full.offset,
            loss=full.loss, strain_weight=full.strain_weight, oob_weight=full.oob_weight,
            shape_weight=full.shape_weight, fixed_mask=reduce(full.mask), interpolation='p1_ac',
            matches=full.matches, match_weight=full.match_weight, strain_model=full.strain_model,
            mind_order=full.mind_order, image_weight=full.image_weight, mind_frame=full.mind_frame)
        item.joint_prior_backend = full.joint_prior_backend
        item.joint_p1_priors = full.joint_p1_priors
        if getattr(args, 'p1_sampling', 'existing') == 'frozen':
            item.prepare_fixed_p1_sampling(args.grid_side, args.grid_side,
                                          dtype=torch.float64, device=device)
        evidences[side] = item
    return evidences, preprocessing, match_metadata


def _record(value, parts):
    return dict(total=float(value), **{key: float(item) for key, item in parts.items()})


def optimize_distortion_budget(args, *, initial_map, fiber_solver=None):
    _validate_configuration(args)
    if fiber_solver is None:
        from qcopt.neural_bijection.dense.coordinated_distortion_budget import optimize_distortion_budget_fiber
        fiber_solver = optimize_distortion_budget_fiber
    args = copy.deepcopy(args)
    args.output = Path(args.output)
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError(args.output)
    if not (len(args.levels) == len(args.image_levels) and args.image_levels[-1] == args.image_side):
        raise ValueError('original aligned image/coefficient schedule required')
    inner = getattr(args, 'inner_steps_by_level', None) or [args.inner_steps] * len(args.levels)
    if len(inner) != len(args.levels) or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in inner):
        raise ValueError('positive per-level maximum gradient budgets required')
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    sync = lambda: torch.cuda.synchronize(device) if device.type == 'cuda' else None
    sync()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with np.load(args.affine, allow_pickle=False) as archive:
        matrix = np.asarray(archive['post_affine_matrix'], dtype=np.float32)
        offset = np.asarray(archive['post_affine_offset'], dtype=np.float32)
    current, initial_map_record = load_incumbent(initial_map, matrix, offset, args.grid_side,
        device=device, minimum_jacobian=args.minimum_jacobian)
    reference = identity_vertices(args.grid_side, device=device).double()
    sync()
    loading_seconds = time.perf_counter() - started
    tick = time.perf_counter()
    evidences, preprocessing, match_metadata = build_pyramid(args, matrix, offset, device)
    objectives = {side: BudgetEvidence(item) for side, item in evidences.items()}
    full = objectives[args.image_side]
    sync()
    feature_seconds = time.perf_counter() - tick
    with torch.no_grad():
        initial, parts = full(current)
        initial_record = _record(initial, parts)
    budget = initial_record['strain']  # EXACT runtime value, fixed for the entire suffix.
    if not np.isfinite(budget) or budget < 0 or not np.isfinite(initial_record['total']):
        raise ValueError('finite initial objective and nonnegative actual ARAP required')
    best_map, best_loss, selected = current.clone(), float(initial), None
    counts, trace, stages = {}, [], []
    numerical_failure = False
    wrapper_evaluations = 1
    tick = time.perf_counter()
    for level, side, limit in zip(args.levels, args.image_levels, inner, strict=True):
        for direction in ((1., 0.), (0., 1.)):
            physical_rms = args.learning_rate * (args.levels[0] - 1) / (level - 1)
            result = fiber_solver(current.detach(), objectives[side], coefficient_side=level,
                direction=direction, distortion_budget=budget, physical_rms_step=physical_rms,
                maximum_gradients=limit, minimum_jacobian=args.minimum_jacobian,
                theta=.95, fraction=.99, armijo=1e-4, max_backtracks=12)
            current = result.vertices.detach()
            with torch.no_grad():
                value, parts = full(current)
                actual_ratio = q1_corner_determinants(current) * (args.grid_side - 1) ** 2
            wrapper_evaluations += 1
            if (not bool(torch.isfinite(current).all() and torch.isfinite(actual_ratio).all()
                    and (actual_ratio > args.minimum_jacobian).all())
                    or not torch.equal(current[:,(0,-1),:,:], reference[:,(0,-1),:,:])
                    or not torch.equal(current[:,:,(0,-1),:], reference[:,:,(0,-1),:])
                    or not np.isfinite(float(value)) or not np.isfinite(float(parts['strain']))
                    or float(parts['strain']) > budget):
                raise RuntimeError('actual stage geometry or fixed ARAP budget failed')
            for key, number in result.counts.items():
                counts[key] = counts.get(key, 0) + number
            stage = dict(level=level, image_side=side, direction=list(direction),
                maximum_gradients=limit, physical_rms_step=physical_rms,
                anchor_total=result.initial_objective, accepted_total=result.final_objective,
                anchor_distortion=result.initial_distortion, accepted_distortion=result.final_distortion,
                accepted_full_total=float(value), accepted_original_full_total=float(parts['original_total']),
                actual_distortion=float(parts['strain']), distortion_budget=budget,
                counts=result.counts, stop_reason=result.stop_reason,
                minimum_contracted_slack=result.minimum_contracted_slack,
                numerical_failure=bool(result.numerical_failure))
            stages.append(stage)
            trace.extend(dict(stage=len(stages)-1, level=level, image_side=side,
                              direction=list(direction), **row) for row in result.trace)
            if float(value) < best_loss:
                best_map, best_loss, selected = current.clone(), float(value), len(stages)-1
            if result.numerical_failure:
                numerical_failure = True
                break
        if numerical_failure:
            break
    sync()
    optimize_seconds = time.perf_counter() - tick
    with torch.no_grad():
        final, parts = full(best_map)
    wrapper_evaluations += 1
    final_record = _record(final, parts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tick = time.perf_counter()
    np.savez(args.output, vertices=best_map.cpu().numpy(), boundary_reference=reference.cpu().numpy(),
        post_affine_matrix=matrix, post_affine_offset=offset, interpolation=np.asarray('p1_ac'))
    serialization_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    certificate = certify_q1_binary_map(args.output)
    certification_seconds = time.perf_counter() - tick
    if not certificate['valid'] or final_record['strain'] > budget or final_record['total'] > initial_record['total']:
        raise RuntimeError('actual selected output certificate/budget/data selection failed')
    report = dict(configuration=_plain(args), optimizer='incumbent_ARAP_budget_spectral_QCQP',
        representation='absolute fine P1ac then original positive affine', landmarks_used=False,
        initial_map=initial_map_record, initial=initial_record, final=final_record,
        distortion_budget=budget, distortion_budget_source='actual runtime R of original fusion300 incumbent ONCE',
        objective='D=image+OOB+1e-4shape+.2fused_points; actual R<=B; original_total is E=D+3R diagnostic',
        gradient_steps=counts.get('gradient_steps', 0), counts=counts,
        objective_evaluations=counts.get('objective_evaluations', 0)+wrapper_evaluations,
        wrapper_objective_evaluations=wrapper_evaluations,
        evaluations=counts.get('trial_evaluations', 0), maximum_gradient_steps=2*sum(inner),
        numerical_failure=numerical_failure, stages=stages, trace=trace,
        selected_stage=selected, selected_stage_scope='None=incoming; otherwise accepted suffix endpoint',
        output_selection='best_full_data_subject_to_incumbent_ARAP_budget',
        schedule_complete=len(stages)==2*len(args.levels) and not numerical_failure,
        budget_mode='distortion_cap', query_count=args.image_side**2, control_vertices=args.grid_side**2,
        saved_binary_certificate=certificate, loading_seconds=loading_seconds, feature_seconds=feature_seconds,
        optimize_seconds=optimize_seconds, serialization_seconds=serialization_seconds,
        certification_seconds=certification_seconds, end_to_end_seconds=time.perf_counter()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
        memory_scope='reset before incumbent loading; includes feature setup, optimizer and output; no matcher resident',
        mind_frame='shared_affine', image_levels=list(args.image_levels), image_preprocessing=preprocessing,
        mind_frame_by_resolution={str(side):item.mind_frame_metadata for side,item in evidences.items()},
        image_match_evidence=match_metadata,
        gradient_scope='at most300 new outer iterations; two separate AD VJPs each; incumbent300 is additional',
        stop_caution='legal retained-map stops are not stationarity; numerical failures remain failed attempts')
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report
