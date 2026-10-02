"""Read-only full-Evidence dispatch equivalence, AFTER first timed applications.

No annotations, optimization, matcher calls, new kernels or production edits.
CPU capture-backend tests are correctness fixtures, never Inductor/GPU evidence.
"""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from tools import coordinated_real_case as app
from tools.coordinated_compiled_priors import make_joint_priors
from tools.coordinated_registration_start import load_incumbent
from tools.digital_q1_dhr_distill import identity_vertices


TOLERANCES = dict(values=dict(rtol=1e-8, atol=1e-10),
                  full_vertex_gradient=dict(rtol=1e-5, atol=1e-8))


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _validate(c, production):
    required = dict(loss='mind', precision='float64', image_precision='float32',
        interpolation='p1_ac', strain_model='p1_arap', mind_frame='shared_affine',
        strain_weight=3., oob_weight=1., shape_weight=1e-4, match_weight=.2,
        match_robust_scale=8.)
    for key, value in required.items():
        if c.get(key) != value:
            raise ValueError(f'current fusion configuration required: {key}')
    for key, default in (('preprocessing','raw_inverted'), ('mind_order','transport'),
        ('image_weight',1.), ('control_hierarchy','fixed'), ('image_objective','continuation')):
        if c.get(key, default) != default:
            raise ValueError(f'unsupported evidence option: {key}')
    if c.get('fixed_mask') is not None or c.get('p1_sampling','existing') not in ('existing','frozen'):
        raise ValueError('original fixed support and declared raster sampler required')
    if production and (c['grid_side'] != 257 or c['image_side'] != 512 or
                       list(c['image_levels']) != [32,64,128,256,512]):
        raise ValueError('production probe requires original257/512 five levels')
    if not c.get('matches') or c['image_side'] not in c['image_levels']:
        raise ValueError('original match table and full raster level required')


def _evidence(c, compiler_backend):
    """Literal production loader/pyramid; B shares only immutable raster features."""
    device = torch.device(c['device']); side = c['grid_side']
    with np.load(c['affine'], allow_pickle=False) as data:
        a = np.asarray(data['post_affine_matrix'], dtype=np.float32)
        b = np.asarray(data['post_affine_offset'], dtype=np.float32)
    if a.shape != (2,2) or b.shape != (2,) or not np.isfinite(a).all() or not np.isfinite(b).all() or np.linalg.det(a.astype(np.float64)) <= 0:
        raise ValueError('finite positive saved affine required')
    fixed, moving, mask, metadata = app.load_registration_evidence(
        Path(c['fixed']), Path(c['moving']), c['image_side'],
        preprocessing=c.get('preprocessing','raw_inverted'), device=device,
        dtype=torch.float32, fixed_mask_path=c.get('fixed_mask'))
    matches = []; match_metadata = []
    for frozen in (False, True):
        item, detail = app.load_image_matches(Path(c['matches']), a, b,
            fixed_path=Path(c['fixed']), moving_path=Path(c['moving']),
            image_side=c['image_side'], device=device, dtype=torch.float64,
            robust_scale=c['match_robust_scale'])
        if frozen:
            detail['fixed_p1_sampling'] = item.prepare_fixed_p1_sampling(side,side,'ac')
        matches.append(item); match_metadata.append(detail)
    matrix = torch.from_numpy(a).to(device=device,dtype=torch.float64)
    offset = torch.from_numpy(b).to(device=device,dtype=torch.float64)
    def create(f,m,w):
        return app.Evidence(f,m,matrix,offset,c['loss'],c['strain_weight'],c['oob_weight'],
            c['shape_weight'], fixed_mask=w, interpolation=c['interpolation'],
            matches=matches[0],match_weight=c['match_weight'],strain_model=c['strain_model'],
            mind_order=c.get('mind_order','transport'),joint_prior_backend='eager',
            image_weight=c.get('image_weight',1.),mind_frame=c['mind_frame'])
    full = create(fixed,moving,mask); eager = {c['image_side']:full}
    for size in sorted(set(c['image_levels'])):
        if size not in eager:
            resize = lambda x: F.interpolate(x,size=(size,size),mode='area')
            eager[size] = create(resize(fixed),resize(moving),resize(full.mask))
    compiled = {}; prior = make_joint_priors(compiler_backend)
    for size, item in eager.items():
        if c.get('p1_sampling','existing') == 'frozen':
            item.prepare_fixed_p1_sampling(side,side,dtype=torch.float64,device=device)
        candidate = copy.copy(item)
        candidate.matches = matches[1]
        candidate.joint_prior_backend = compiler_backend
        candidate.joint_p1_priors = prior
        compiled[size] = candidate
    return eager, compiled, a, b, dict(preprocessing=metadata,matches=match_metadata)


def _evaluate(evidence, vertices):
    variable = vertices.detach().clone().requires_grad_(True)
    _sync(variable.device); started = time.perf_counter()
    total, parts = evidence(variable)
    gradient, = torch.autograd.grad(total,variable)
    _sync(variable.device); elapsed = time.perf_counter()-started
    names = ['total', *sorted(parts)]
    values = torch.stack([total.detach(),*[parts[k].detach() for k in sorted(parts)]]).double().cpu()
    return dict(names=names, values=values, gradient=gradient.detach().double().cpu(),
                seconds=elapsed)


def _compare(left,right):
    if left['names'] != right['names']:
        raise ValueError('objective component names changed')
    result = {}
    def finite_number(value):
        number=float(value)
        return number if math.isfinite(number) else None
    for key, field in (('values','values'),('full_vertex_gradient','gradient')):
        a,b = left[field],right[field]; finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        tolerance = TOLERANCES[key]
        delta = b-a
        result[key] = dict(finite=finite,
            passed=finite and bool(torch.allclose(a,b,**tolerance)),
            maximum_absolute_error=finite_number(delta.abs().max()) if finite else None,
            rms_error=finite_number(delta.square().mean().sqrt()) if finite else None,
            relative_l2_error=finite_number(delta.norm()/a.norm().clamp_min(1e-30)) if finite else None)
    result['passed'] = all(result[key]['passed'] for key in TOLERANCES)
    return result


def _check(configuration, anchor_path, *, outputpath, production, compiler_backend):
    outputpath = Path(outputpath)
    if outputpath.exists():
        raise FileExistsError(outputpath)
    c = dict(configuration); _validate(c,production)
    if production and compiler_backend != 'inductor':
        raise ValueError('production requires actual Inductor')
    torch.set_num_threads(c.get('threads',2)); device=torch.device(c['device'])
    report = dict(passed=False, status='running',annotations_read=False,
        compiler_backend=compiler_backend,production=production,torch_version=torch.__version__,
        scope='full production Evidence forward and full vertex VJP, no optimization; call only after first timed A/B registrations',
        cache_scope='uses caller process/compiler caches; setup/probes may warm them; not a cold compilation benchmark',
        tolerance_basis='fixed before probing: double-valued components use1e-8/1e-10; gradients allow float32 raster-backward accumulation with1e-5/1e-8. Both same-backend repeats must also pass; noise is reported, never used to enlarge tolerances. These are numerical comparison bounds, not an error theorem.',
        tolerances=TOLERANCES,order=['eager','compiled','compiled','eager'],rows=[])
    started=time.perf_counter(); outputpath.parent.mkdir(parents=True,exist_ok=True)
    try:
        eager,compiled,a,b,metadata=_evidence(c,compiler_backend)
        anchor,anchor_metadata=load_incumbent(anchor_path,a,b,c['grid_side'],device=device,
            minimum_jacobian=c.get('minimum_jacobian',.001))
        identity=identity_vertices(c['grid_side'],device=device).double()
        _sync(device); report.update(setup_seconds=time.perf_counter()-started,
            evidence_metadata=metadata,anchor=anchor_metadata)
        probe_started=time.perf_counter()
        for name,vertices in (('identity',identity),('saved_deformed',anchor)):
            for size in c['image_levels']:
                outputs=[_evaluate(items[size],vertices) for items in (eager,compiled,compiled,eager)]
                comparisons={f'{i}_vs_{j}':_compare(outputs[i],outputs[j])
                             for i,j in ((0,3),(1,2),(0,1),(0,2),(3,1),(3,2))}
                finite=all(torch.isfinite(item['values']).all().item() for item in outputs)
                report['rows'].append(dict(state=name,image_side=size,
                    passed=all(item['passed'] for item in comparisons.values()),
                    component_names=outputs[0]['names'],
                    values=[item['values'].tolist() for item in outputs] if finite else None,
                    call_seconds=[item['seconds'] for item in outputs],comparisons=comparisons))
        _sync(device); report['probe_seconds']=time.perf_counter()-probe_started
        report.update(passed=all(row['passed'] for row in report['rows']),status='complete')
    except Exception as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}',passed=False)
    report['total_seconds']=time.perf_counter()-started
    outputpath.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return report


def check(configuration:dict, anchor_path:Path, *, outputpath:Path)->dict:
    return _check(configuration,anchor_path,outputpath=outputpath,
                  production=True,compiler_backend='inductor')
