"""Bounded positive-conductance inverse diagnostic on a fixed target map."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import scipy.optimize
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
import torch

from phase6_fit_dense_map_oracle import target_map
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.tutte.symmetric import MatrixFreeSymmetricTutteLayer


class TimeLimit(Exception):
    pass


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--side",type=int,required=True)
    parser.add_argument("--target-kind",choices=("base","high32"),default="high32")
    parser.add_argument("--lower",type=float,default=1.0)
    parser.add_argument("--upper",type=float,default=16.0)
    parser.add_argument("--maxiter",type=int,default=1000)
    parser.add_argument("--time-limit",type=float,default=300.0)
    args=parser.parse_args()
    if args.side<3 or not 0<args.lower<args.upper:
        raise ValueError("invalid grid or conductance bounds")
    started=time.perf_counter()
    mesh=structured_rectangle(args.side-1,args.side-1)
    topology=MatrixFreeSymmetricTutteLayer(mesh)
    edges=np.asarray(topology.active_edges,dtype=np.int64)
    interior=np.asarray(topology.interior_vertices,dtype=np.int64)
    target=target_map(args.side,torch.device("cpu"),torch.float64,
                      args.target_kind)[0].reshape(-1,2).numpy()
    index=np.full(mesh.n_vertices,-1,dtype=np.int64)
    index[interior]=np.arange(len(interior))
    a,b=edges[:,0],edges[:,1]
    displacement=target[a]-target[b]
    edge_ids=np.arange(len(edges))
    rows,cols,values=[],[],[]
    for endpoint,sign in ((a,1),(b,-1)):
        valid=index[endpoint]>=0
        for coordinate in range(2):
            rows.append(2*index[endpoint[valid]]+coordinate)
            cols.append(edge_ids[valid])
            values.append(sign*displacement[valid,coordinate])
    matrix=sparse.coo_matrix(
        (np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),
        shape=(2*len(interior),len(edges))).tocsr()
    scaled=matrix*((args.side-1)**2)
    assembly_seconds=time.perf_counter()-started
    initial=np.full(len(edges),4.0,dtype=np.float64)
    initial=np.clip(initial,args.lower,args.upper)
    records=[]
    iteration=0
    last=initial.copy()

    def objective(weights):
        residual=scaled@weights
        return 0.5*float(residual@residual),scaled.T@residual

    initial_loss=objective(initial)[0]
    def callback(weights):
        nonlocal iteration,last
        iteration+=1
        last=weights.copy()
        elapsed=time.perf_counter()-started
        if iteration==1 or iteration%25==0:
            residual=matrix@weights
            records.append({
                "iteration":iteration,
                "elapsed_seconds":elapsed,
                "equilibrium_l2":float(np.linalg.norm(residual)),
                "equilibrium_max_abs":float(np.max(np.abs(residual))),
            })
        if elapsed>args.time_limit:
            raise TimeLimit()

    try:
        result=scipy.optimize.minimize(
            objective,initial,jac=True,method="L-BFGS-B",
            bounds=scipy.optimize.Bounds(args.lower,args.upper),
            callback=callback,
            options={"maxiter":args.maxiter,"maxls":30,
                     "ftol":1e-16,"gtol":1e-10},
        )
        weights=result.x
        status=str(result.message)
        success=bool(result.success)
        function_evaluations=int(result.nfev)
    except TimeLimit:
        weights=last
        status="time_limit_at_callback"
        success=False
        function_evaluations=None

    residual=matrix@weights
    ia,ib=index[a],index[b]
    diagonal=np.bincount(ia[ia>=0],weights=weights[ia>=0],minlength=len(interior))
    diagonal+=np.bincount(ib[ib>=0],weights=weights[ib>=0],minlength=len(interior))
    both=(ia>=0)&(ib>=0)
    operator=sparse.diags(diagonal).tocsr()
    operator+=sparse.coo_matrix(
        (np.concatenate((-weights[both],-weights[both])),
         (np.concatenate((ia[both],ib[both])),
          np.concatenate((ib[both],ia[both])))),
        shape=(len(interior),len(interior))).tocsr()
    rhs=np.zeros((len(interior),2),dtype=np.float64)
    ab=(ia>=0)&(ib<0)
    ba=(ib>=0)&(ia<0)
    np.add.at(rhs,ia[ab],weights[ab,None]*target[b[ab]])
    np.add.at(rhs,ib[ba],weights[ba,None]*target[a[ba]])
    recovered=sparse_linalg.spsolve(operator,rhs)
    complete=target.copy()
    complete[interior]=recovered
    face=mesh.faces
    points=complete[face]
    signed=(points[:,1,0]-points[:,0,0])*(points[:,2,1]-points[:,0,1])-(
        points[:,1,1]-points[:,0,1])*(points[:,2,0]-points[:,0,0])
    print(json.dumps({
        "method":"bounded_positive_conductance_least_squares_diagnostic",
        "side":args.side,"target_kind":args.target_kind,
        "control_vertices":mesh.n_vertices,"control_faces":mesh.n_faces,
        "active_edges":len(edges),"constraint_rows":matrix.shape[0],
        "conductance_bounds":[args.lower,args.upper],
        "initial_objective":initial_loss,
        "final_objective":objective(weights)[0],
        "optimizer_success":success,"optimizer_status":status,
        "iterations":iteration,"function_evaluations":function_evaluations,
        "minimum_conductance":float(weights.min()),
        "maximum_conductance":float(weights.max()),
        "equilibrium_l2_residual":float(np.linalg.norm(residual)),
        "equilibrium_max_abs_residual":float(np.max(np.abs(residual))),
        "direct_recovery_max_coordinate_error":float(
            np.max(np.abs(recovered-target[interior]))),
        "direct_recovery_rmse":float(
            np.sqrt(np.mean((recovered-target[interior])**2))),
        "direct_recovery_minimum_signed_area_ratio":float(
            signed.min()*(args.side-1)**2),
        "assembly_seconds":assembly_seconds,
        "total_seconds":time.perf_counter()-started,
        "records":records,
    },sort_keys=True,separators=(",",":")))


if __name__=="__main__":
    main()
