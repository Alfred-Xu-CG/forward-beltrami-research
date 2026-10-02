"""Native STANDARD DHR with the frozen accepted512 shared initializer only.

This is a modified-initialization counterfactual, NOT the unmodified published
pipeline or a matched-objective/topology ablation. No labels/matches are read.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from tools.coordinated_dhr_released_baseline import configure, _save
from tools.coordinated_joint_pose_batch import source_cases
from tools.coordinated_terminal_detail_inputs import input_pair, native_cases


def _frames(layouts, padding, preprocessed_hw):
    wh=np.asarray(preprocessed_hw,dtype=float)[::-1]
    ratio=float(padding['initial_resample_ratio']) if padding.get('initial_resampling',False) else 1.
    if wh.shape!=(2,) or not np.isfinite(wh).all() or np.any(wh<2) or not np.isfinite(ratio) or ratio<=0:
        raise ValueError('finite nondegenerate native preprocessing frame required')
    result={}
    for role,key,index in [('fixed','target',2),('moving','source',1)]:
        scale=np.asarray(layouts[role]['effective_original_to_canvas_scale_xy'],dtype=float)
        pad=np.asarray(layouts[role]['padding_xy'],dtype=float)
        loader=float(padding[key+'_resample_ratio'])
        dhr_pad=np.asarray([padding[f'pad_{index}'][1][0],padding[f'pad_{index}'][0][0]],dtype=float)
        if (scale.shape!=(2,) or pad.shape!=(2,) or not all(np.isfinite(x).all() for x in (scale,pad,dhr_pad,[loader]))
                or np.any(scale<=0) or np.any(pad<0) or np.any(dhr_pad<0) or loader<=0):
            raise ValueError('positive exact layout/loader scales and nonnegative padding required')
        c=np.eye(3);c[:2,:2]=np.diag(scale/512);c[:2,2]=(.5*scale+pad)/512
        n=np.eye(3);n[:2,:2]=np.diag(2*loader/(ratio*wh));n[:2,2]=2*(.5*loader+dhr_pad)/(ratio*wh)-1
        result[role]=(c,n)
    return result


def shared_affine_to_theta(matrix,offset,layouts,padding,preprocessed_hw):
    """Column-homogeneous N_m C_m^-1 H(A,b) C_f N_f^-1; float64."""
    a,b=np.asarray(matrix,dtype=float),np.asarray(offset,dtype=float)
    if a.shape!=(2,2) or b.shape!=(2,) or not np.isfinite(a).all() or not np.isfinite(b).all() or np.linalg.det(a)<=0:
        raise ValueError('finite orientation-preserving 2x2 shared affine and length2 offset required')
    frames=_frames(layouts,padding,preprocessed_hw)
    cf,nf=frames['fixed'];cm,nm=frames['moving']
    h=np.eye(3);h[:2,:2]=a;h[:2,2]=b
    return (nm@np.linalg.inv(cm)@h@cf@np.linalg.inv(nf))[:2]


def initial_field_audit(field,theta64,layouts,padding,*,chunk_rows=64):
    """All native preprocessed nodes against analytic theta64, bounded CPU chunks.

    Checks the actual released float32 initial field, including casting/grid
    roundoff. Does not promise affine extrapolation by border-clamped sampling.
    """
    if field.ndim!=4 or field.shape[0]!=1 or field.shape[-1]!=2 or chunk_rows<1:
        raise ValueError('single HxWx2 displacement field and positive chunk size required')
    h,w=field.shape[1:3]
    cm,nm=_frames(layouts,padding,(h,w))['moving']
    to_canvas=(cm@np.linalg.inv(nm))[:2,:2]*512
    maximum=canvas_maximum=0.
    for start in range(0,h,chunk_rows):
        y,x=np.meshgrid(np.arange(start,min(start+chunk_rows,h)),np.arange(w),indexing='ij')
        z=np.stack([2*(x+.5)/w-1,2*(y+.5)/h-1],axis=-1)
        expected=z@np.asarray(theta64)[:,:2].T+np.asarray(theta64)[:,2]-z
        actual=field[0,start:start+chunk_rows].detach().cpu().double().numpy()
        if not np.isfinite(actual).all():raise ValueError('nonfinite actual initial displacement field')
        error=actual-expected
        maximum=max(maximum,float(np.max(np.abs(error))))
        canvas_maximum=max(canvas_maximum,float(np.max(np.linalg.norm(error@to_canvas.T,axis=-1))))
    return dict(preprocessed_nodes=h*w,maximum_normalized_error=maximum,
        maximum_canvas_pixel_error=canvas_maximum,
        check='all preprocessed lattice nodes, literal float32 field versus float64 analytic conjugate',
        dense_extrapolation_outside_lattice_guaranteed=False,
        boundary_caveat='released dense-field border clamping is not affine extrapolation outside the native lattice')


def shared_pipeline_class(base_class,transform_to_field):
    """Override only native initialization; inherited preprocessing/nonrigid/save."""
    class SharedInitialPipeline(base_class):
        def run_initial_registration(self):
            if not self.registration_parameters['run_initial_registration']:
                raise ValueError('initial stage must remain enabled')
            if self.pre_source.shape!=self.pre_target.shape:
                raise ValueError('common native preprocessed frame required')
            tick=time.perf_counter()
            a,b,layouts=self.shared_initialization
            theta=shared_affine_to_theta(a,b,layouts,self.padding_params,self.pre_source.shape[-2:])
            self.initial_transform=torch.as_tensor(theta,dtype=torch.float32,device=self.pre_source.device)[None]
            self.initial_displacement_field=transform_to_field(self.initial_transform,self.pre_source.size())
            self.current_displacement_field=self.initial_displacement_field
            if self.pre_source.is_cuda:torch.cuda.synchronize()
            self.initial_registration_time=time.perf_counter()-tick
            audit_tick=time.perf_counter()
            self.shared_initial_audit=initial_field_audit(self.initial_displacement_field,theta,layouts,self.padding_params)
            self.shared_initial_audit['accepted512_query_audit']=canvas_query_audit(
                self.initial_displacement_field,theta,a,b,layouts,self.padding_params)
            self.shared_initial_audit['audit_seconds']=time.perf_counter()-audit_tick
            self.shared_theta64=theta.tolist()
    return SharedInitialPipeline


def canvas_query_audit(field,theta64,matrix,offset,layouts,padding):
    """All512 pixel centers: analytic conjugacy and literal border sampling."""
    import torch.nn.functional as F
    h,w=field.shape[1:3]
    frames=_frames(layouts,padding,(h,w));cf,nf=frames['fixed'];cm,nm=frames['moving']
    to_native=nf@np.linalg.inv(cf);to_canvas=cm@np.linalg.inv(nm)
    y,x=np.meshgrid(np.arange(512),np.arange(512),indexing='ij')
    q=np.stack([(x+.5)/512,(y+.5)/512],axis=-1)
    z=q@to_native[:2,:2].T+to_native[:2,2]
    expected=q@np.asarray(matrix).T+np.asarray(offset)
    def mapped(t):
        value=z@t[:,:2].T+t[:,2]
        return value@to_canvas[:2,:2].T+to_canvas[:2,2]
    exact_error=np.linalg.norm((mapped(np.asarray(theta64))-expected)*512,axis=-1)
    cast_error=np.linalg.norm((mapped(np.asarray(theta64,dtype=np.float32).astype(float))-expected)*512,axis=-1)
    grid=torch.as_tensor(z,dtype=field.dtype,device=field.device)[None]
    sampled=F.grid_sample(field.permute(0,3,1,2),grid,mode='bilinear',padding_mode='border',align_corners=False)
    sample_z=sampled[0].permute(1,2,0).detach().cpu().double().numpy()+z
    dense_q=sample_z@to_canvas[:2,:2].T+to_canvas[:2,2]
    dense_error=np.linalg.norm((dense_q-expected)*512,axis=-1)
    inside=(np.abs(z[...,0])<=1-1/w)&(np.abs(z[...,1])<=1-1/h)
    return dict(query_count=512*512,analytic_theta64_max_canvas_pixels=float(exact_error.max()),
        cast_theta32_max_canvas_pixels=float(cast_error.max()),
        native_lattice_inside_count=int(inside.sum()),native_lattice_outside_count=int((~inside).sum()),
        literal_border_sample_max_canvas_pixels=float(dense_error.max()),
        literal_border_sample_inside_max_canvas_pixels=float(dense_error[inside].max()) if inside.any() else None,
        literal_border_sample_outside_max_canvas_pixels=float(dense_error[~inside].max()) if (~inside).any() else None,
        no_queries_dropped=True,query_definition='all512 accepted-canvas pixel centers, including letterbox')


def prepare_cases(args):
    cases=source_cases(args.fusion_predictions)
    native=native_cases(args.native22_inputs)
    rows=[]
    for case in cases:
        sources,layouts=input_pair(case,args.layouts,native,args.miit_source)
        for role in ('fixed','moving'):
            layout=layouts[role]
            wh=np.asarray(layout['original_wh']);resized=np.asarray(layout['resized_wh'])
            scale=np.asarray(layout['effective_original_to_canvas_scale_xy']);pad=np.asarray(layout['padding_xy'])
            if (any(v.shape!=(2,) for v in (wh,resized,scale,pad))
                    or not all(np.isfinite(v).all() for v in (wh,resized,scale,pad))
                    or any(not np.equal(v,np.floor(v)).all() for v in (wh,resized,pad))
                    or np.any(wh<=0) or np.any(resized<=0) or np.any(pad<0) or np.any(pad+resized>512)
                    or not np.array_equal(scale,resized/wh)):
                raise ValueError('exact integer original accepted512 resize/pad layout required')
        affine=Path(case['original_configuration']['affine']).resolve()
        with np.load(affine,allow_pickle=False) as saved:
            a=np.array(saved['post_affine_matrix'],dtype=float);b=np.array(saved['post_affine_offset'],dtype=float)
        if a.shape!=(2,2) or b.shape!=(2,) or not np.isfinite(a).all() or not np.isfinite(b).all() or np.linalg.det(a)<=0:
            raise ValueError('positive frozen SG affine required')
        row=dict(name=case['name'],cohort=case['cohort'],status='pending',
            fixed=str(sources['fixed']),moving=str(sources['moving']),shared_affine_source=str(affine),
            shared_affine_matrix=a.tolist(),shared_affine_offset=b.tolist(),accepted512_layouts=layouts)
        if case['cohort']=='miit':
            _,m,_,f=case['name'].split('_');row.update(moving_section=int(m),fixed_section=int(f))
        rows.append(row)
    return rows


def run(args,*,dhr=None,transform_to_field=None):
    if args.device not in ('cpu','cuda') or isinstance(args.threads,bool) or args.threads<1:
        raise ValueError('cpu/cuda device and positive threads required')
    output=Path(args.output).resolve()
    if output.exists():raise FileExistsError(output)
    rows=prepare_cases(args)
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(args.threads)
    import_tick=time.perf_counter()
    if dhr is None:import deeperhistreg as dhr
    if transform_to_field is None:
        # Exactly the helper used in the installed pipeline, not a custom rasterizer.
        import sys
        transform_to_field=sys.modules[dhr.direct_registration.DeeperHistReg_FullResolution.__module__].w.tc_transform_to_tc_df
    import_seconds=time.perf_counter()-import_tick
    pipeline_class=shared_pipeline_class(dhr.direct_registration.DeeperHistReg_FullResolution,transform_to_field)
    base=dhr.configs.default_initial_nonrigid()
    output.mkdir(parents=True)
    _save(output/'released_preset.json',base)
    try:version=importlib.metadata.version('deeperhistreg')
    except importlib.metadata.PackageNotFoundError:version='not available'
    report=dict(preset='default_initial_nonrigid',package_version=version,
        package_source=str(getattr(dhr,'__file__','unknown')),package_import_seconds=import_seconds,
        started_utc=datetime.now(timezone.utc).isoformat(),prediction_complete=False,annotations_read=False,
        rows=rows,pair_denominator=len(rows),source_fusion_predictions=str(Path(args.fusion_predictions).resolve()),
        initialization='frozen SG accepted512 affine conjugated to native frame; released initializer replaced',
        protocol='supporting native STANDARD modified-initialization counterfactual, not unmodified published pipeline',
        algorithm_changes=['initial-registration stage replaced with the existing shared affine'],
        unchanged='native source images; full STANDARD preprocessing, nonrigid pyramid, NCC, diffusion, optimizer, interpolation, composition, export',
        io_adaptation='same PIL native ratio1 and field-only export as native own-initializer baseline',
        cost_scope='native pipeline plus chunked initial-field audit; prior SG initializer computation excluded, not zero-cost',
        topology='no hard topology guarantee; not an isolated topology/objective/preprocessing/resolution ablation',
        map_direction='fixed native pixel centers to moving native pixel centers',
        initial_field_storage='theta and audit metadata only; no second full initial-field export')
    started=time.perf_counter()
    def persist():
        report['elapsed_seconds']=time.perf_counter()-started
        _save(output/'predictions.json',report)
    persist()
    for row in rows:
        folder=output/row['name'];folder.mkdir()
        params=configure(base,device=args.device,output=folder)
        _save(folder/'config.json',params);row['configuration']=f"{row['name']}/config.json"
        tick=time.perf_counter();pipeline=None;peak_reset=False
        try:
            for role in ('fixed','moving'):
                with Image.open(row[role]) as im:
                    if im.mode not in ('RGB','L') or im.getexif().get(274,1)!=1:
                        raise ValueError('native L/RGB identity orientation required')
                    if list(im.size)!=row['accepted512_layouts'][role]['original_wh']:
                        raise ValueError('native dimensions differ from frozen accepted512 layout')
                    row[role+'_original_wh']=list(im.size);row[role+'_mode']=im.mode
                    row[role+'_file_bytes']=Path(row[role]).stat().st_size
            if args.device=='cuda':
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();peak_reset=True
                row['cuda_free_bytes_before_case'],row['cuda_total_bytes']=torch.cuda.mem_get_info()
            pipeline=pipeline_class(params)
            pipeline.shared_initialization=(row['shared_affine_matrix'],row['shared_affine_offset'],row['accepted512_layouts'])
            pipeline.run_registration(row['moving'],row['fixed'],str(folder))
            if args.device=='cuda':torch.cuda.synchronize()
            final=folder/'released_dhr/Results_Final'
            for key,name in [('field','displacement_field.mha'),('postprocessing_params','postprocessing_params.json')]:
                path=final/name
                if not path.is_file():raise FileNotFoundError(path)
                row[key]=path.relative_to(output).as_posix()
            row.update(status='ok',preprocessed_shape=list(pipeline.pre_source.shape),
                field_shape=list(pipeline.current_displacement_field.shape),loaded_padded_shape=list(pipeline.source.shape),
                initial_transform=pipeline.initial_transform.detach().cpu().tolist(),initial_transform_finite=True,
                initial_transform_frame='preprocessed normalized [-1,1] target-to-source affine; align_corners=False',
                initial_transform_determinant=float(torch.linalg.det(pipeline.initial_transform[0,:,:2].double())),
                conjugate_theta64=pipeline.shared_theta64,initial_field_audit=pipeline.shared_initial_audit,
                native_padding_params=pipeline.padding_params,preprocessing_seconds=pipeline.preprocessing_time,
                initial_seconds=pipeline.initial_registration_time,nonrigid_seconds=pipeline.nonrigid_registration_time,
                registration_seconds=pipeline.total_registration_time)
        except Exception as error:row.update(status='failed',error=f'{type(error).__name__}: {error}')
        finally:
            row['complete_call_seconds']=time.perf_counter()-tick
            row['peak_allocated_bytes']=torch.cuda.max_memory_allocated() if peak_reset else None
            del pipeline
            persist()
    report.update(prediction_complete=True,all25_terminal=True,successful_pairs=sum(r['status']=='ok' for r in rows))
    persist()
    # Existing scorer inputs appear only after every one of the25 attempts.
    for cohort,name in [('miit','miit_predictions.json'),('existing','existing_predictions.json')]:
        selected=[r for r in rows if (r['cohort']=='miit')==(cohort=='miit')]
        _save(output/name,{**report,'rows':selected,'pair_denominator':len(selected),
            'successful_pairs':sum(r['status']=='ok' for r in selected)})
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('fusion-predictions','layouts','native22-inputs','miit-source','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--threads',type=int,default=2)
    report=run(parser.parse_args())
    print(json.dumps({k:report[k] for k in ('successful_pairs','pair_denominator','elapsed_seconds')}))


if __name__=='__main__':main()
