"""Read-only native DHR map audit on its pixel-center rectangle, without padding."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image


def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def exact_orientation(a,b,c):
    ratios = [float(x).as_integer_ratio() for p in (a,b,c) for x in p]
    power = max(den.bit_length()-1 for _,den in ratios)
    values = [num << (power-den.bit_length()+1) for num,den in ratios]
    ax,ay,bx,by,cx,cy = values
    determinant = (bx-ax)*(cy-ay)-(by-ay)*(cx-ax)
    return int(determinant > 0)-int(determinant < 0)


def exact_boundary_check(vertices):
    """Exact segment predicates for the already constructed binary64 boundary."""
    boundary = np.concatenate((vertices[0], vertices[1:, -1],
                               vertices[-1, -2::-1], vertices[-2:0:-1, 0]))
    if not np.isfinite(boundary).all():
        return {"simple": False, "reason": "nonfinite boundary"}
    ratios = [[float(v).as_integer_ratio() for v in p] for p in boundary]
    power = max(den.bit_length() - 1 for point in ratios for _, den in point)
    exact = [tuple(num << (power - den.bit_length() + 1) for num, den in point)
             for point in ratios]
    count = len(exact)

    def orient(a, b, c):
        return ((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))

    def on_segment(a, b, p):
        return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and
                min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))

    zero_edges = sum(exact[i] == exact[(i+1) % count] for i in range(count))
    adjacent_overlaps = 0
    for i in range(count):
        a, b, c = exact[i-1], exact[i], exact[(i+1) % count]
        if orient(a, b, c) == 0:
            adjacent_overlaps += ((a[0]-b[0])*(c[0]-b[0]) +
                                  (a[1]-b[1])*(c[1]-b[1])) > 0
    following = np.roll(boundary, -1, axis=0)
    lower, upper = np.minimum(boundary, following), np.maximum(boundary, following)
    intersections = []
    intersection_count = 0
    for i in range(count):
        candidates = np.flatnonzero(np.all(upper[i] >= lower, axis=1) &
                                    np.all(upper >= lower[i], axis=1))
        for j in candidates:
            j = int(j)
            if j <= i+1 or (i == 0 and j == count-1):
                continue
            a, b = exact[i], exact[(i+1) % count]
            c, d = exact[j], exact[(j+1) % count]
            v1, v2, v3, v4 = orient(a,b,c), orient(a,b,d), orient(c,d,a), orient(c,d,b)
            hit = ((v1*v2 < 0 and v3*v4 < 0) or
                   (v1 == 0 and on_segment(a,b,c)) or
                   (v2 == 0 and on_segment(a,b,d)) or
                   (v3 == 0 and on_segment(c,d,a)) or
                   (v4 == 0 and on_segment(c,d,b)))
            if hit:
                intersection_count += 1
                if len(intersections) < 10:
                    intersections.append([i, j])
    area2 = sum(exact[i][0]*exact[(i+1) % count][1] -
                exact[i][1]*exact[(i+1) % count][0] for i in range(count))
    return dict(simple=not (zero_edges or adjacent_overlaps or intersection_count),
                vertex_count=count, zero_edges=int(zero_edges),
                adjacent_overlaps=int(adjacent_overlaps),
                nonadjacent_intersections=intersection_count,
                first_intersecting_edge_pairs=intersections,
                signed_area_sign=int(area2 > 0)-int(area2 < 0),
                predicates="exact integer arithmetic on binary64 map nodes; float64 bounding boxes only prune disjoint pairs",
                scope="boundary of the trimmed pixel-center domain only")


def stratified_signs(corners, mask):
    """Separate face counts and cell counts; zero and negative cells may overlap."""
    result={}
    for name,indices in (('all_four',[0,1,2,3]),('p1_ac',[1,3]),('p1_bd',[0,2])):
        selected=corners[...,indices][mask]
        result[name]=dict(selected_cells=int(mask.sum()),tested_corners=int(selected.size),
            negative_corners=int((selected<0).sum()),zero_corners=int((selected==0).sum()),
            negative_cells=int(np.any(selected<0,axis=-1).sum()),
            zero_cells=int(np.any(selected==0,axis=-1).sum()),
            nonpositive_cells=int(np.any(selected<=0,axis=-1).sum()),
            minimum_normalized_corner=float(selected.min()) if selected.size else None)
    return result


def audit(path, fixed_path):
    params = json.loads((path.parent / 'postprocessing_params.json').read_text())
    if any(params[key] != [[0,0],[0,0]] for key in ('pad_1','pad_2')):
        raise ValueError('Only zero-padding cases covered')
    if any(params.get(key, 1) != 1 for key in
           ('source_resample_ratio','target_resample_ratio','initial_resample_ratio')):
        raise ValueError('Only unit-scale cases covered')
    raw = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    if raw.shape != (2,512,512) or raw.dtype != np.float32:
        raise ValueError(f'Unexpected native field {raw.shape} {raw.dtype}')
    row, col = np.mgrid[:512,:512]
    reference = np.stack((col+.5, row+.5), axis=-1) / 512
    vertices = reference + raw.astype(np.float64).transpose(1,2,0) / 512
    with Image.open(fixed_path) as image:
        if image.size != (512,512):
            raise ValueError('Original supplied fixed raster must be 512 square; no resampling')
        gray=np.asarray(image.convert('L'),dtype=np.uint8)
    tissue=1-gray.astype(np.float64)/255 > .04
    tissue_corners=np.stack((tissue[:-1,:-1],tissue[:-1,1:],tissue[1:,1:],tissue[1:,:-1]),axis=-1)
    tissue_any=np.any(tissue_corners,axis=-1)
    tissue_all=np.all(tissue_corners,axis=-1)
    a,b,c,d = vertices[:-1,:-1],vertices[:-1,1:],vertices[1:,1:],vertices[1:,:-1]
    corner = np.stack((cross(b-a,d-a), cross(b-a,c-a),
                       cross(b-d,c-d), cross(c-a,d-a)), axis=-1) * 512**2
    finite = np.isfinite(corner)
    triangle_arrays=((a,b,d),(a,b,c),(d,b,c),(a,c,d))
    minimum_index=tuple(int(x) for x in np.unravel_index(np.argmin(corner),corner.shape))
    ri,ci,ki=minimum_index
    minimum_exact_sign=exact_orientation(*(v[ri,ci] for v in triangle_arrays[ki]))
    # Audit cancellation-sensitive cases only; no tolerance changes the reported signs.
    near_indices=np.argwhere(np.abs(corner)<1e-10)
    exact_near_counts={str(sign):0 for sign in (-1,0,1)}
    near_disagreements=0
    for ri,ci,ki in near_indices:
        sign=exact_orientation(*(v[ri,ci] for v in triangle_arrays[ki]))
        exact_near_counts[str(sign)]+=1
        near_disagreements+=sign != int(np.sign(corner[ri,ci,ki]))
    cell_row,cell_col=np.mgrid[:511,:511]
    distance=np.minimum.reduce((cell_row,cell_col,510-cell_row,510-cell_col))
    strata={}
    for band in (0,1,4,16,32):
        interior=distance>=band
        strata[str(band)]={name:stratified_signs(corner,interior & mask) for name,mask in
            (('all_cells',np.ones_like(interior)),('tissue_any',tissue_any),('tissue_all',tissue_all))}
    ri,ci,ki=minimum_index
    node_indices=((ri,ci),(ri,ci+1),(ri+1,ci+1),(ri+1,ci))
    fixture=dict(cell_row=ri,cell_column=ci,corner_id=ki,
        corner_triangle_labels=[['a','b','d'],['a','b','c'],['d','b','c'],['a','c','d']][ki],
        normalized_determinant=float(corner[minimum_index]),reference_doubled_area=1/512**2,
        nodes={label:dict(row=int(r),column=int(c),reference_unit=reference[r,c].tolist(),
                         mapped_unit=vertices[r,c].tolist(),
                         mapped_unit_float_hex=[float(x).hex() for x in vertices[r,c]],
                         saved_pixel_displacement=raw[:,r,c].astype(np.float64).tolist(),
                         fixed_gray_uint8=int(gray[r,c]),static_tissue=bool(tissue[r,c]))
               for label,(r,c) in zip(('a','b','c','d'),node_indices)})
    return dict(field=str(path), dtype=str(raw.dtype), control_vertices=512**2,
                cells=511**2, corner_determinants=int(corner.size),
                source_domain=[.5/512, 1-.5/512],
                formula="Y[i,j]=((j+.5+field[0,i,j])/512,(i+.5+field[1,i,j])/512)",
                arithmetic="float32 saved pixel displacement promoted to float64, then added to exact dyadic center reference",
                minimum_normalized_corner=float(corner[finite].min()),
                maximum_normalized_corner=float(corner[finite].max()),
                negative=int((corner < 0).sum()), zero=int((corner == 0).sum()),
                nonfinite=int((~finite).sum()),
                cells_with_nonpositive_corner=int((np.any(corner <= 0,axis=-1)).sum()),
                per_corner_negative=[int((corner[...,k] < 0).sum()) for k in range(4)],
                minimum_location_row_column_corner=list(minimum_index),
                minimum_exact_binary64_sign=minimum_exact_sign,
                near_zero_exact_audit=dict(candidate_threshold_normalized=1e-10,
                    candidate_count=len(near_indices),exact_sign_counts=exact_near_counts,
                    sign_disagreements_with_float64=int(near_disagreements)),
                stratification=dict(fixed_raster=str(fixed_path),
                    tissue_rule='1 - original fixed PNG convert(L)/255 > 0.04; no warp, no current overlap, no resizing',
                    tissue_any_rule='OR over four original fixed pixels at the cell vertices',
                    tissue_all_rule='AND over the same four original fixed pixels',
                    band_rule='Retain source cells with min(row,column,510-row,510-column)>=band; no extension',
                    bands=strata),minimum_fixture=fixture,
                corner_order=['abd','abc','dbc','acd'],
                boundary=exact_boundary_check(vertices),
                scope="Saved native 512-center interpolant only; no full unit-square claim, no algorithm-level topology guarantee, no resampling or endpoint extension")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--canvas',type=Path,default=Path('D:/QC_optimization_data/digital_topology_wsi/birl_anhir_dev/canvas'))
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    names=['dhr_histo_common','dhr_lesions_common','dhr_rat_kidney_common',
           'known_dhr_wide_shear','known_dhr_local_rotation','known_dhr_coarse_fine']
    rows={}
    for name in names:
        fixed_path=(args.canvas/(name.removeprefix('dhr_').removesuffix('_common')+'_fixed512.png')
                    if name.startswith('dhr_') else
                    args.base/('known_image257_matrix_'+name.removeprefix('known_dhr_')+'_fixed.png'))
        rows[name]=audit(args.base/name/'common_affine_dhr/Results_Final/displacement_field.mha',fixed_path)
    report=dict(question='Are the actual native center-node maps four-corner positive with simple boundary?',
                saved_field_units='pixels; tc_df_to_np_df multiplies normalized tensor displacement by size/2 before MHA saving',
                method='four oriented triangle determinants divided by reference doubled area (1/512)^2',
                results=rows)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({name:{key:row[key] for key in ('minimum_normalized_corner','maximum_normalized_corner','negative','zero','nonfinite','boundary')}
                      for name,row in rows.items()},indent=2))


if __name__=='__main__':
    main()
