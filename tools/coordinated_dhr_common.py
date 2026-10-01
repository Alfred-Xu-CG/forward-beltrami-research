"""Native DHR nonrigid optimization from the exact shared canvas affine.

The image preprocessing/native NCC/regularizer remain DHR's, so this is an
application baseline, not an isolated shared-objective geometry ablation.
Only identical unpadded square canvas inputs without initial resampling are
supported here, avoiding an unverified affine frame conversion.
"""
import argparse
import json
from pathlib import Path
import time

import deeperhistreg
import numpy as np
import torch
import torch.nn.functional as F

from tools.digital_dhr_baseline import configure_device


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.affine) as archive:
        matrix = np.asarray(archive["post_affine_matrix"], dtype=np.float32)
        offset = np.asarray(archive["post_affine_offset"], dtype=np.float32)
    if matrix.shape != (2,2) or offset.shape != (2,) or np.linalg.det(matrix.astype(np.float64)) <= 0:
        raise ValueError("positive affine required")
    class CommonAffinePipeline(deeperhistreg.direct_registration.DeeperHistReg_FullResolution):
        def run_initial_registration(self):
            if self.pre_source.shape[-2:] != (512,512) or self.pre_target.shape[-2:] != (512,512):
                raise ValueError("512 square shared canvases required")
            if self.padding_params.get("initial_resample_ratio",1) != 1:
                raise ValueError("initial resampling would change affine frame")
            for key in ("pad_1","pad_2"):
                if any(value != 0 for pair in self.padding_params[key] for value in pair):
                    raise ValueError("unpadded shared input canvases required")
            a = torch.as_tensor(matrix, device=self.pre_source.device)
            b = torch.as_tensor(offset, device=self.pre_source.device)
            theta = torch.cat((a,(2*b+a@torch.ones(2,device=a.device)-1)[:,None]),dim=1)[None]
            identity = torch.tensor([[[1.,0.,0.],[0.,1.,0.]]],device=a.device)
            self.initial_transform = theta
            self.initial_displacement_field = F.affine_grid(theta,self.pre_source.size(),align_corners=False)-F.affine_grid(identity,self.pre_source.size(),align_corners=False)
            self.current_displacement_field = self.initial_displacement_field
            self.initial_registration_time = 0.
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    params = deeperhistreg.configs.default_initial_nonrigid_fast()
    configure_device(params,args.device)
    params.update(case_name="common_affine_dhr",logging_path=str(args.output/"deeperhistreg.log"))
    params["loading_params"].update(loader="pil",source_resample_ratio=1.,target_resample_ratio=1.)
    params["saving_params"]["final_saver"] = "pil"
    params["save_final_images"] = False
    params["save_final_displacement_field"] = True
    params["preprocessing_params"].update(initial_resolution=768,save_results=False)
    params["initial_registration_params"]["save_results"] = False
    params["nonrigid_registration_params"].update(save_results=False,registration_size=512,
        num_levels=5,used_levels=5,iterations=[30]*5,learning_rates=[.005,.0025,.0025,.0025,.0025],alphas=[1.5]*5)
    (args.output/"config.json").write_text(json.dumps(params,indent=2,default=str)+"\n")
    start = time.perf_counter()
    pipeline = CommonAffinePipeline(params)
    pipeline.run_registration(str(args.moving),str(args.fixed),str(args.output))
    report = dict(runtime_seconds=time.perf_counter()-start,nonrigid_seconds=pipeline.nonrigid_registration_time,
                  preprocessing_seconds=pipeline.preprocessing_time,common_affine=str(args.affine),
                  initialization="exact supplied unit-canvas affine, no new matcher or nonrigid teacher",
                  native_objective="DHR preprocessing/NCC/regularizer, not shared geometry objective",
                  post_affine_matrix=matrix.tolist(),post_affine_offset=offset.tolist(),
                  field_shape=list(pipeline.current_displacement_field.shape))
    (args.output/"runtime.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("fixed","moving","affine","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--threads",type=int,default=2)
    run(p.parse_args())


if __name__ == "__main__":
    main()
