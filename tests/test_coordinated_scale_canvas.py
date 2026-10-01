import importlib.util
import json

import pytest
import torch  # Load before NumPy in the local native-runtime environment.
import numpy as np
from PIL import Image


def api():
    name="tools.coordinated_scale_canvas"
    assert importlib.util.find_spec(name) is not None,"exact-frame scale prep not implemented"
    from tools.coordinated_scale_canvas import prepare_scaled_canvases
    return prepare_scaled_canvases


def fixture(tmp_path):
    canvas=tmp_path/"old";matches=tmp_path/"matches";output=tmp_path/"new"
    canvas.mkdir();matches.mkdir()
    layout=dict(side=16,scale_assumption="same assumed source-pixel scale",landmarks_used=False)
    inputs=[]
    for role,wh,resized,pad in (("moving",(16,24),(8,12),(4,2)),("fixed",(20,32),(10,16),(3,0))):
        source=tmp_path/(role+".jpg")
        y,x=np.indices((wh[1],wh[0]))
        rgb=np.stack(((11*x+3*y)%180,(7*x+5*y)%190,(2*x+13*y)%200),-1).astype(np.uint8)
        Image.fromarray(rgb).save(source,quality=95)
        old_png=canvas/("histo_"+role+"16.png")
        # Deliberately unrelated old raster proves source JPEGs are used.
        Image.new("RGB",(16,16),(253,252,251)).save(old_png)
        layout[role]=dict(source=str(source),original_wh=list(wh),canvas_png=str(old_png),
            resized_wh=list(resized),padding_xy=list(pad),canvas_mpp=2.,original_mpp_xy=[1.,1.],
            effective_original_to_canvas_scale_xy=[resized[0]/wh[0],resized[1]/wh[1]],
            original_center_to_canvas_center="canvas_xy=(original_xy+0.5)*effective_scale_xy-0.5+padding_xy")
        inputs.extend((source,old_png))
    layout_path=canvas/"histo_layout.json";layout_path.write_text(json.dumps(layout))
    matrix=np.array([[.95,.07],[-.04,1.02]],dtype=np.float32)
    offset=np.array([.01,-.02],dtype=np.float32)
    affine=canvas/"histo_initial_affine.npz"
    np.savez_compressed(affine,post_affine_matrix=matrix,post_affine_offset=offset,unused_constant=np.array([17],dtype=np.int16))
    points=np.linspace(.2,.7,16).reshape(8,2)
    record=dict(fixed=layout['fixed']['canvas_png'],moving=layout['moving']['canvas_png'],affine=str(affine),
        post_affine_matrix=matrix.tolist(),post_affine_offset=offset.tolist(),image_side=16,
        source_points_unit=points.tolist(),target_points_unit=(points+.013).tolist(),confidence=np.linspace(.3,.8,8).tolist(),
        raw_matches=8,targets_manual_landmarks_or_dense_teacher_loaded=False,method="existing frozen image-only matcher")
    match_path=matches/"histo_common_sg_raw_matches.json";match_path.write_text(json.dumps(record))
    inputs.extend((layout_path,affine,match_path))
    return canvas,matches,output,layout,record,inputs


def test_direct_original_pixels_exact_frame_affine_matches_and_preserved_inputs(tmp_path):
    canvas,matches,output,old,record,inputs=fixture(tmp_path)
    before={path:path.read_bytes() for path in inputs}
    api()(canvas,matches,output,cases=["histo"],factor=2)
    new=json.loads((output/"histo_layout.json").read_text())
    assert new['side']==32 and new['landmarks_used'] is False
    for role in ("moving","fixed"):
        previous=old[role];current=new[role]
        assert current['source']==previous['source'] and current['original_wh']==previous['original_wh']
        assert current['resized_wh']==[2*v for v in previous['resized_wh']]
        assert current['padding_xy']==[2*v for v in previous['padding_xy']]
        assert current['canvas_mpp']==previous['canvas_mpp']/2
        np.testing.assert_array_equal(current['effective_original_to_canvas_scale_xy'],2*np.asarray(previous['effective_original_to_canvas_scale_xy']))
        wh=np.asarray(previous['original_wh'])
        points=np.vstack(([-.5,-.5],wh-.5,[0.,0.],wh/2,wh-1))
        old_unit=((points+.5)*previous['effective_original_to_canvas_scale_xy']+previous['padding_xy'])/16
        new_unit=((points+.5)*current['effective_original_to_canvas_scale_xy']+current['padding_xy'])/32
        np.testing.assert_array_equal(old_unit,new_unit)
        with Image.open(previous['source']) as image:
            expected=image.convert('RGB').resize(tuple(current['resized_wh']),Image.Resampling.BILINEAR)
        with Image.open(current['canvas_png']) as image:
            assert image.size==(32,32) and image.mode=='RGB'
            raster=np.asarray(image)
        px,py=current['padding_xy'];rw,rh=current['resized_wh']
        np.testing.assert_array_equal(raster[py:py+rh,px:px+rw],np.asarray(expected))
        background=np.ones((32,32),dtype=bool);background[py:py+rh,px:px+rw]=False
        assert (raster[background]==255).all()
    assert (output/'histo_initial_affine.npz').read_bytes()==before[canvas/'histo_initial_affine.npz']
    new_record=json.loads((output/'histo_common_sg_raw_matches.json').read_text())
    for key in ('source_points_unit','target_points_unit','confidence','post_affine_matrix','post_affine_offset'):
        assert new_record[key]==record[key]
    assert new_record['image_side']==32 and new_record['prediction_side']==16
    assert new_record['origin_match_record']==str((matches/'histo_common_sg_raw_matches.json').resolve())
    assert 'not fresh' in new_record['transport_method']
    assert new_record['physical_robust_scale_factor']==2
    assert new_record['fixed']==new['fixed']['canvas_png'] and new_record['moving']==new['moving']['canvas_png']
    assert {path:path.read_bytes() for path in inputs}==before


@pytest.mark.parametrize('case',['factor_zero','factor_float','factor_bool','cases_string','cases_empty','cases_duplicate','unknown_case',
    'wrong_dimensions','wrong_scale','bad_size_type','upsampling','existing','match_side','match_side_type','match_count_type',
    'prediction_side_type','match_affine','match_raster','match_points','match_confidence','match_provenance','inverted_affine','path_type','missing_second_case'])
def test_preflight_guards_no_partial_outputs(tmp_path,case):
    canvas,matches,output,layout,record,_=fixture(tmp_path)
    factor=2;cases=['histo'];expected=(ValueError,FileNotFoundError)
    if case=='factor_zero':factor=0
    elif case=='factor_float':factor=2.
    elif case=='factor_bool':factor=True
    elif case=='cases_string':cases='histo'
    elif case=='cases_empty':cases=[]
    elif case=='cases_duplicate':cases=['histo','histo']
    elif case=='unknown_case':cases=['unknown']
    elif case=='wrong_dimensions':
        layout['moving']['original_wh'][0]+=1
        layout['moving']['effective_original_to_canvas_scale_xy'][0]=layout['moving']['resized_wh'][0]/layout['moving']['original_wh'][0]
    elif case=='wrong_scale':layout['moving']['effective_original_to_canvas_scale_xy'][0]+=.01
    elif case=='bad_size_type':layout['side']=True
    elif case=='upsampling':factor=3
    elif case=='existing':
        output.mkdir();(output/'histo_layout.json').write_text('existing material');expected=(FileExistsError,)
    elif case=='match_side':record['image_side']=32
    elif case=='match_side_type':record['image_side']=16.
    elif case=='match_count_type':record['raw_matches']=8.
    elif case=='prediction_side_type':record['prediction_side']=True
    elif case=='match_affine':record['post_affine_matrix'][0][0]+=.01
    elif case=='match_raster':record['fixed']='unrelated.png'
    elif case=='match_points':record['target_points_unit'][0][0]=1.01
    elif case=='match_confidence':record['confidence'][0]=-1
    elif case=='match_provenance':record['targets_manual_landmarks_or_dense_teacher_loaded']=True
    elif case=='inverted_affine':
        matrix=np.array([[-1.,0.],[0.,1.]],dtype=np.float32)
        np.savez(canvas/'histo_initial_affine.npz',post_affine_matrix=matrix,post_affine_offset=np.zeros(2,dtype=np.float32))
        record['post_affine_matrix']=matrix.tolist();record['post_affine_offset']=[0.,0.]
    elif case=='path_type':canvas=12
    elif case=='missing_second_case':cases=['histo','rat_kidney']
    if isinstance(canvas,type(tmp_path)):
        (canvas/'histo_layout.json').write_text(json.dumps(layout))
    (matches/'histo_common_sg_raw_matches.json').write_text(json.dumps(record))
    with pytest.raises(expected):api()(canvas,matches,output,cases=cases,factor=factor)
    if case=='existing':
        assert (output/'histo_layout.json').read_text()=='existing material'
        assert len(list(output.iterdir()))==1
    else:assert not output.exists() or not list(output.iterdir())
