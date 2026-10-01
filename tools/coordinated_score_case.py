"""Separate read-only development landmark scoring; never called by optimizer."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from tools.digital_birl_landmark_score import original_pixel_to_canvas_unit, canvas_unit_to_original_pixel, q1_at_queries
from tools.digital_dhr_field_eval import _landmarks
from tools.digital_q1_real_eval import load_effective_vertices


def p1_at_queries_numpy(vertices,points,diagonal):
    """Independent barycentric matrix solves, not the Torch weight formulas."""
    rows,columns=vertices.shape[:2]
    x,y=points[:,0]*(columns-1),points[:,1]*(rows-1)
    column=np.floor(x).astype(int).clip(0,columns-2)
    row=np.floor(y).astype(int).clip(0,rows-2)
    local=np.stack((x-column,y-row),-1)
    cell=np.stack((vertices[row,column],vertices[row,column+1],vertices[row+1,column+1],vertices[row+1,column]),1)
    reference=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    if diagonal=="ac":
        ids=np.where((local[:,1]<=local[:,0])[:,None],np.array([0,1,2]),np.array([0,2,3]))
    elif diagonal=="bd":
        ids=np.where((local.sum(-1)<=1)[:,None],np.array([0,1,3]),np.array([1,2,3]))
    else:
        raise ValueError("diagonal must be ac or bd")
    source=reference[ids]
    matrix=np.stack((source[:,1]-source[:,0],source[:,2]-source[:,0]),axis=-1)
    coordinate=np.linalg.solve(matrix,(local-source[:,0])[...,None])[...,0]
    weights=np.concatenate((1-coordinate.sum(-1,keepdims=True),coordinate),-1)
    values=np.take_along_axis(cell,ids[...,None],axis=1)
    return (weights[...,None]*values).sum(1)


def affine_domain_bound(expected,matrix,offset,side):
    """Necessary image-domain bound only; never an optimizer input."""
    latent=np.linalg.solve(matrix,(expected-offset).T).T
    outside=((latent<0)|(latent>1)).any(-1)
    polygon=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])@matrix.T+offset
    distances=[]
    for k in range(4):
        start,end=polygon[k],polygon[(k+1)%4]
        vector=end-start
        t=((expected-start)@vector/(vector@vector)).clip(0,1)
        closest=start+t[:,None]*vector
        distances.append(np.linalg.norm(expected-closest,axis=-1))
    bound=np.where(outside,np.min(distances,axis=0)*side,0.)
    return dict(outside_required_landmarks=int(outside.sum()),landmark_count=len(expected),
                mean_unavoidable_canvas_px=float(bound.mean()),max_unavoidable_canvas_px=float(bound.max()),
                scope="necessary range bound from fixed affine image of unit rectangle; not sufficient representation criterion")


def score(layout_path, fixed_path, moving_path, maps, affine, output, dhr=None):
    if output.exists():
        raise FileExistsError(output)
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    loaded = {name: load_effective_vertices(path) for name,path in maps.items()}
    interpolation={}
    for name,path in maps.items():
        with np.load(path) as archive:
            interpolation[name]=str(archive["interpolation"].item()) if "interpolation" in archive else "q1"
    if not all(report["composite_representation_valid"] for _,report in loaded.values()):
        raise ValueError("invalid saved map")
    with Image.open(layout["fixed"]["source"]) as image:
        fixed_size = image.size
    with Image.open(layout["moving"]["source"]) as image:
        moving_size = image.size
    fixed = _landmarks(fixed_path,fixed_size)
    moving = _landmarks(moving_path,moving_size)
    ids = sorted(fixed.keys() & moving.keys())
    if not ids:
        raise ValueError("no shared landmark IDs")
    f = np.stack([fixed[key] for key in ids]); m = np.stack([moving[key] for key in ids])
    query = original_pixel_to_canvas_unit(f,layout["fixed"],layout["side"])
    expected = original_pixel_to_canvas_unit(m,layout["moving"],layout["side"])
    predicted={}
    for name,(vertices,_) in loaded.items():
        mode=interpolation[name]
        if mode=="q1":
            predicted[name]=q1_at_queries(vertices,query)
        elif mode in ("p1_ac","p1_bd"):
            predicted[name]=p1_at_queries_numpy(vertices,query,mode[-2:])
        else:
            raise ValueError("unsupported saved interpolation: "+mode)
    with np.load(affine) as data:
        matrix=data["post_affine_matrix"].astype(np.float64)
        offset=data["post_affine_offset"].astype(np.float64)
        predicted["common_affine"] = query @ matrix.T+offset
        domain_bound=affine_domain_bound(expected,matrix,offset,layout["side"])
    if dhr:
        import SimpleITK as sitk
        import torch
        from tools.digital_compare_appearance import dhr_map_at_unit_queries
        for name,field_path,params_path in dhr:
            field = sitk.GetArrayFromImage(sitk.ReadImage(str(field_path)))
            params = json.loads(Path(params_path).read_text(encoding="utf-8"))
            mapped = dhr_map_at_unit_queries(field,params,fixed_size=(512,512),moving_size=(512,512),
                query=torch.tensor(query,dtype=torch.float32).reshape(1,1,-1,2))
            predicted[name] = mapped[0,0].numpy().astype(np.float64)
    results = {}
    for name,p in predicted.items():
        canvas_error = np.linalg.norm((p-expected)*layout["side"],axis=-1)
        native = np.linalg.norm(canvas_unit_to_original_pixel(p,layout["moving"],layout["side"])-m,axis=-1)
        results[name] = dict(mean_canvas_px=float(canvas_error.mean()),p90_canvas_px=float(np.percentile(canvas_error,90)),
                             p95_canvas_px=float(np.percentile(canvas_error,95)),max_canvas_px=float(canvas_error.max()),
                             mean_native_moving_px=float(native.mean()),per_landmark_canvas_px=dict(zip(ids,canvas_error.tolist())))
    report = dict(protocol="reused public development specimen, not independent or official ACROBAT/ANHIR test",
                  interpolation=interpolation,map_direction="fixed to moving",landmark_count=len(ids),
                  fixed_only_ids=sorted(fixed.keys()-moving.keys()),moving_only_ids=sorted(moving.keys()-fixed.keys()),
                  layout=str(layout_path),fixed_landmarks=str(fixed_path),moving_landmarks=str(moving_path),
                  maps={name:str(path) for name,path in maps.items()},affine=str(affine),results=results)
    report["fixed_boundary_target_domain"]=domain_bound
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("layout","fixed_landmarks","moving_landmarks","affine","output"):
        parser.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    parser.add_argument("--map",action="append",required=True,help="name=absolute NPZ path")
    parser.add_argument("--dhr",action="append",nargs=3,metavar=("NAME","FIELD","PARAMS"),
                        help="native DHR field on the SAME supplied 512 canvases; no topology certificate")
    args = parser.parse_args()
    maps = dict(item.split("=",1) for item in args.map)
    result = score(args.layout,args.fixed_landmarks,args.moving_landmarks,
                   {name:Path(path) for name,path in maps.items()},args.affine,args.output,args.dhr)
    print(json.dumps({name:{key:value for key,value in values.items() if key != "per_landmark_canvas_px"}
                      for name,values in result["results"].items()}))


if __name__ == "__main__":
    main()
