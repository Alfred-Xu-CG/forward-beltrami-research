"""Shared ground-truth legality suite, not a fitting-accuracy benchmark.

Run with PYTHONPATH=src, MKL_THREADING_LAYER=SEQUENTIAL, OMP_NUM_THREADS=1:
  python experiments/phase5/route1_validity_suite.py --json validity.json

One JSON and one compact CSV, no per-run directories. Defaults cover control
sides 11/17/25/33/49, all eight analytic maps, nine positive-Tutte targets per
mesh, and all four images at 256/512. Targets are CPU float64 DirectTutte
references; image sampling is audited at float32 by default. No target is
passed as its own accuracy reference. No inverse or inverse-consistency is
constructed. Failed geometry is explicitly ineligible as ground truth.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np
import scipy
import torch

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.benchmarks import (
    ANALYTIC_DEFORMATION_NAMES, IMAGE_NAMES, identity_query_grid,
    make_registration_pair, sample_analytic_deformation, synthetic_image,
    warp_image_backward,
)
from qcopt.neural_bijection.metrics import compute_p1_map_metrics
from qcopt.neural_bijection.tutte.direct import DirectTutteLayer
from qcopt.neural_bijection.tutte.instance_optimization import build_directed_target


@dataclass(frozen=True)
class ValidityConfig:
    controls: tuple[int, ...] = (11, 17, 25, 33, 49)
    strengths: tuple[float, ...] = (0.25, 1.0, 3.0)
    seeds: tuple[int, ...] = (1701, 1702, 1703)
    resolutions: tuple[int, ...] = (256, 512)
    image_dtype: str = 'float32'
    threads: int = 1


def audit_map(mesh, mapped):
    """Recompute global P1 metrics, independent edge-area/Jacobian and |mu|.

    This does not consume cached generator certificates. The numerical
    global audit is provided by the shared injectivity checker, independently
    of the target's harmonic solve. It is not an exact-predicate certificate.
    """
    mapped = np.asarray(mapped, dtype=np.float64)
    metrics = compute_p1_map_metrics(mesh, mapped)  # deliberately NO target=
    source = mesh.vertices[mesh.faces]
    target = mapped[mesh.faces]
    ds = np.stack((source[:, 1]-source[:, 0], source[:, 2]-source[:, 0]), axis=-1)
    dt = np.stack((target[:, 1]-target[:, 0], target[:, 2]-target[:, 0]), axis=-1)
    source_det = ds[:, 0, 0]*ds[:, 1, 1]-ds[:, 0, 1]*ds[:, 1, 0]
    target_det = dt[:, 0, 0]*dt[:, 1, 1]-dt[:, 0, 1]*dt[:, 1, 0]
    if not metrics.global_injectivity_certificate or not np.all(target_det > 0):
        raise ValueError('independent P1 topology/positive-area audit failed')
    jacobian = dt @ np.linalg.inv(ds)
    a,b,c,d = (jacobian[:,0,0],jacobian[:,0,1],jacobian[:,1,0],jacobian[:,1,1])
    fz = (a+d+1j*(c-b))/2
    fbar = (a-d+1j*(c+b))/2
    mu_abs = np.abs(fbar/fz)
    if not np.all(np.isfinite(mu_abs)) or not np.all(mu_abs < 1):
        raise ValueError('represented face Beltrami magnitudes must be finite and below one')
    metric_names = ('flip_count','minimum_signed_area','minimum_area_ratio',
                    'boundary_order_min_gap','global_injectivity_certificate')
    return dict(metrics={name:getattr(metrics,name) for name in metric_names},
                independent_minimum_signed_area=float(target_det.min()/2),
                independent_minimum_area_ratio=float((target_det/source_det).min()),
                mu_magnitude=dict(min=float(mu_abs.min()),median=float(np.median(mu_abs)),
                                  p95=float(np.quantile(mu_abs,.95)),max=float(mu_abs.max())),
                unit_square_sampling_compatible=bool(np.all((mapped>=0)&(mapped<=1))),
                accuracy=None, inverse_consistency=None)


def _bilinear_oracle(image, coordinates):
    """Independent NumPy 4-corner interpolation, for in-range (x,y) only."""
    values = image.detach().double().cpu().numpy()[0,0]
    q = coordinates.detach().double().cpu().numpy()
    if not np.all(np.isfinite(q)) or not np.all((q>=0)&(q<=1)):
        raise ValueError('oracle coordinates must be finite and in the unit square')
    height,width = values.shape
    x,y = q[...,0]*(width-1),q[...,1]*(height-1)
    x0,y0 = np.floor(x).astype(int),np.floor(y).astype(int)
    x1,y1 = np.minimum(x0+1,width-1),np.minimum(y0+1,height-1)
    tx,ty = x-x0,y-y0
    return ((1-tx)*(1-ty)*values[y0,x0]+tx*(1-ty)*values[y0,x1]
            +(1-tx)*ty*values[y1,x0]+tx*ty*values[y1,x1])


def audit_image(name, resolution, *, dtype=torch.float32):
    image = synthetic_image(name,height=resolution,width=resolution,dtype=dtype)
    repeated = synthetic_image(name,height=resolution,width=resolution,dtype=dtype)
    finite = bool(torch.isfinite(image).all())
    deterministic = bool(torch.equal(image,repeated))
    minimum,maximum = float(image.min()),float(image.max())
    if not finite or not deterministic or minimum!=0 or maximum!=1:
        raise ValueError('image must be deterministic, finite, nonconstant and normalized to [0,1]')
    grid = identity_query_grid(resolution,resolution,dtype=dtype)
    tolerance = 8*torch.finfo(dtype).eps*resolution
    identity_error = float((warp_image_backward(image,grid)-image).abs().max())
    # Exactly the next moving-image pixel: positive x means sampling to the
    # right, NOT translating the moving image's content right. Crop the fixed
    # domain so every sample remains in range; no padding can hide a mismatch.
    shifted_grid = grid[:,1:,:].clone()
    shift_error = float((warp_image_backward(image,shifted_grid)-image[:,:,:,1:]).abs().max())
    if not np.isfinite([identity_error,shift_error]).all() or identity_error>tolerance or shift_error>tolerance:
        raise ValueError('identity/positive-x backward shift convention failed')
    pair = make_registration_pair(name,deformation='boundary_sliding',
                                  height=resolution,width=resolution,dtype=dtype)
    q = pair.backward_map
    x = q[0,:,0]
    y = q[:,0,1]
    # This particular dense target is a product map (g(x),y). Strict order,
    # exact corners, and no cross dependence certify its sampled rectangular
    # P1 map without relying on the fixed image to validate the target.
    legal = (bool(torch.isfinite(q).all()) and bool(((q>=0)&(q<=1)).all())
             and bool((x[1:]>x[:-1]).all()) and bool((y[1:]>y[:-1]).all())
             and float(x[0])==0 and float(x[-1])==1 and float(y[0])==0 and float(y[-1])==1
             and bool(torch.equal(q[...,0],x.expand(resolution,-1)))
             and bool(torch.equal(q[...,1],y[:,None].expand(-1,resolution))))
    if not legal:
        raise ValueError('registration ground truth failed independent in-range/order audit')
    oracle = _bilinear_oracle(pair.moving,q)
    pair_error = float(np.max(np.abs(pair.fixed.double().cpu().numpy()[0,0]-oracle)))
    if not bool(torch.isfinite(pair.fixed).all()) or not np.isfinite(pair_error) or pair_error>tolerance:
        raise ValueError('registration pair failed independent bilinear backward convention')
    return dict(kind='image',name=name,resolution=resolution,status='ok',eligible_ground_truth=True,
                image_dtype=str(dtype).removeprefix('torch.'),image_finite=finite,
                image_min=minimum,image_max=maximum,deterministic=deterministic,
                identity_max_error=identity_error,positive_x_shift_max_error=shift_error,
                pair_bilinear_oracle_max_error=pair_error,sampling_tolerance=tolerance,
                backward_convention='fixed_to_moving_align_corners_true',
                pair_deformation='boundary_sliding',pair_ground_truth_legal=True,
                pair_minimum_x_gap=float((x[1:]-x[:-1]).min()),
                accuracy=None,inverse_consistency=None)


def _validate(config):
    for name in ('controls','strengths','seeds','resolutions'):
        values = getattr(config,name)
        if not values or len(values)!=len(set(values)):
            raise ValueError(f'{name} must be nonempty and unique')
    if any(type(n) is not int or n<3 for n in config.controls):
        raise ValueError('control sides must be integers >=3')
    if any(type(n) is not int or n<2 for n in config.resolutions):
        raise ValueError('image resolutions must be integers >=2')
    if any(type(n) is not int or n<0 for n in config.seeds):
        raise ValueError('seeds must be nonnegative integers')
    if any(not np.isfinite(n) or n<=0 for n in config.strengths):
        raise ValueError('strengths must be finite and positive')
    if config.image_dtype not in ('float32','float64') or type(config.threads) is not int or config.threads<1:
        raise ValueError('use float32/float64 images and a positive thread count')


def run_suite(config=ValidityConfig()):
    _validate(config)
    started=perf_counter()
    rows=[]
    def record(base, operation):
        row=dict(base,status='failure',eligible_ground_truth=False,accuracy=None,inverse_consistency=None)
        try:
            row.update(operation())
            row.update(status='ok',eligible_ground_truth=True)
        except Exception as error:
            row['error']=dict(type=type(error).__name__,message=str(error))
        rows.append(row)
    previous_threads=torch.get_num_threads()
    torch.set_num_threads(config.threads)
    try:
        for side in config.controls:
            mesh=structured_rectangle(side-1,side-1)
            for name in ANALYTIC_DEFORMATION_NAMES:
                def analytic():
                    sample=sample_analytic_deformation(mesh,name)
                    bound=sample.continuous_determinant_lower_bound
                    if not np.isfinite(bound) or bound<=0:
                        raise ValueError('analytic continuous determinant bound must be positive')
                    return dict(audit_map(mesh,sample.vertices),continuous_determinant_lower_bound=bound)
                record(dict(kind='analytic',name=name,control=side),analytic)
            system=DirectTutteLayer(mesh).system
            mask=torch.tensor(system.valid_mask.copy())
            for strength in config.strengths:
                for seed in config.seeds:
                    def random_target():
                        # Only control geometry is needed. A 2x2 query table
                        # avoids conflating legality with dense-grid workloads.
                        target=build_directed_target(mesh,image_height=2,image_width=2,
                                                     seed=seed,strength=strength,height=1.)
                        probabilities=torch.softmax(target.interior_logits.masked_fill(~mask,-torch.inf),dim=-1)
                        supported=probabilities[mask]
                        if not bool(torch.isfinite(supported).all()) or not bool((supported>0).all()):
                            raise ValueError('represented supported probabilities must be strictly positive')
                        return dict(audit_map(mesh,target.control.numpy()),
                                    supported_probability_min=float(supported.min()),
                                    supported_probability_max=float(supported.max()),
                                    probability_row_sum_max_error=float((probabilities.sum(-1)-1).abs().max()))
                    record(dict(kind='positive_tutte',name='directed_random',control=side,
                                seed=seed,strength=strength,height=1.,target_dtype='float64'),random_target)
        for resolution in config.resolutions:
            for name in IMAGE_NAMES:
                record(dict(kind='image',name=name,resolution=resolution),
                       lambda:audit_image(name,resolution,dtype=getattr(torch,config.image_dtype)))
    finally:
        torch.set_num_threads(previous_threads)
    try:
        repository=Path(__file__).resolve().parents[2]
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repository,text=True,stderr=subprocess.DEVNULL).strip()
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=repository,text=True,
                                           stderr=subprocess.DEVNULL).strip())
    except (OSError,subprocess.CalledProcessError):
        commit=dirty=None
    return dict(status='ok' if all(row['status']=='ok' for row in rows) else 'failure',
                config=asdict(config),rows=rows,wall_seconds=perf_counter()-started,
                environment=dict(host=platform.node(),python=platform.python_version(),
                                 numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,
                                 commit=commit,git_dirty=dirty),
                semantics=dict(geometry='continuous determinant lower bounds and numerical sampled-P1 legality are separate; per-face mu is recomputed from edge Jacobians',
                               random='CPU float64 directed positive-Tutte targets at H=1; strengths are logit scales, not prescribed distortion or condition numbers',
                               accuracy='not evaluated: no self-map or self-mu zero errors; only actual fitted outputs may be compared with an audited target',
                               domain='a topologically valid analytic map can leave the unit square and be unsuitable for the in-range registration sampler',
                               images='identity and positive-x pixel shift plus independent NumPy bilinear oracle; boundary-sliding dense target audited before oracle use',
                               inverse_consistency='not evaluated: no inverse map is constructed'))


def write_results(report,json_path,csv_path=None):
    json_path=Path(json_path)
    csv_path=Path(csv_path) if csv_path is not None else json_path.with_suffix('.csv')
    if json_path.resolve()==csv_path.resolve():
        raise ValueError('JSON and CSV paths must differ')
    text=json.dumps(report,indent=2,allow_nan=False)
    json_path.parent.mkdir(parents=True,exist_ok=True)
    csv_path.parent.mkdir(parents=True,exist_ok=True)
    json_path.write_text(text+'\n',encoding='utf-8')
    fields=['kind','name','control','seed','strength','resolution','status','eligible_ground_truth',
            'continuous_determinant_lower_bound','independent_minimum_signed_area','independent_minimum_area_ratio',
            'flip_count','global_injectivity_certificate','mu_p95','mu_max','supported_probability_min',
            'supported_probability_max','unit_square_sampling_compatible','image_min','image_max',
            'identity_max_error','positive_x_shift_max_error','pair_bilinear_oracle_max_error',
            'sampling_tolerance','error']
    with csv_path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for receipt in report['rows']:
            row={key:receipt.get(key) for key in fields}
            row.update({key:receipt.get('metrics',{}).get(key) for key in ('flip_count','global_injectivity_certificate')})
            row.update(mu_p95=receipt.get('mu_magnitude',{}).get('p95'),mu_max=receipt.get('mu_magnitude',{}).get('max'),
                       error=(receipt.get('error') or {}).get('message'))
            writer.writerow(row)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    defaults=ValidityConfig()
    for field in ('controls','strengths','seeds','resolutions'):
        parser.add_argument('--'+field,nargs='+',type=float if field=='strengths' else int,default=getattr(defaults,field))
    parser.add_argument('--image-dtype',choices=['float32','float64'],default=defaults.image_dtype)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--json',type=Path,required=True)
    parser.add_argument('--csv',type=Path)
    args=parser.parse_args(argv)
    config=ValidityConfig(**{field:tuple(getattr(args,field)) for field in ('controls','strengths','seeds','resolutions')},
                          image_dtype=args.image_dtype,threads=args.threads)
    report=run_suite(config)
    write_results(report,args.json,args.csv)
    print(json.dumps(dict(status=report['status'],rows=len(report['rows']),wall_seconds=report['wall_seconds'])))
    return 0 if report['status']=='ok' else 2


if __name__=='__main__':
    raise SystemExit(main())
