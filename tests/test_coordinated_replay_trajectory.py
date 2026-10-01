import numpy as np
import pytest

from tools.coordinated_replay_trajectory import score_arrays


@pytest.mark.parametrize("diagonal",["ac","bd"])
def test_prefix_selection_uses_objective_not_target_error(diagonal):
    axis=np.linspace(0,1,5)
    y,x=np.meshgrid(axis,axis,indexing="ij");identity=np.stack((x,y),-1)
    maps=np.stack((identity+[.02,0],identity+[.05,0],identity+[.01,0]))
    q=np.array([[.31,.42],[.63,.73]])
    matrix=np.array([[1.2,.1],[-.1,.9]]);offset=np.array([.02,.01])
    target=(q+[.05,0])@matrix.T+offset
    result=score_arrays(maps,q,target,matrix,offset,512,[.7,.9,.8],1.,[1.,2.,3.],diagonal)
    assert result["stages"][1]["accepted"]["mean_canvas_px"]<1e-12
    assert all(row["selected_prefix_stage"]==0 for row in result["stages"])
    assert result["stages"][1]["image_selected_prefix"]["mean_canvas_px"]>10
    assert all(row["accepted"]["required_landmarks"]==2 for row in result["stages"])


def test_initial_map_can_remain_image_selected():
    axis=np.linspace(0,1,3);y,x=np.meshgrid(axis,axis,indexing="ij")
    identity=np.stack((x,y),-1)
    q=np.array([[.4,.4],[.6,.7]])
    result=score_arrays(np.stack((identity+[.02,0],)),q,q,np.eye(2),np.zeros(2),512,[2.],1.,[1.],"ac")
    assert result["stages"][0]["selected_prefix_stage"] is None
    assert result["stages"][0]["image_selected_prefix"]["mean_canvas_px"]==0
    with pytest.raises(ValueError):
        score_arrays([],q,q,np.eye(2),np.zeros(2),512,[2.],1.,[],"ac")


def test_tiny_record_never_loads_landmarks_and_saves_one_archive(tmp_path):
    import json
    from PIL import Image
    from tools.coordinated_replay_trajectory import record
    image=np.random.default_rng(55).integers(30,210,(16,16),dtype=np.uint8)
    fixed,moving=tmp_path/"fixed.png",tmp_path/"moving.png"
    Image.fromarray(image).save(fixed);Image.fromarray(np.roll(image,1,axis=1)).save(moving)
    affine=tmp_path/"affine.npz"
    np.savez(affine,post_affine_matrix=np.eye(2),post_affine_offset=np.zeros(2))
    parameters=dict(fixed=str(fixed),moving=str(moving),affine=str(affine),output=str(tmp_path/"old.npz"),
        grid_side=9,image_side=16,image_levels=[8,16],levels=[5,9],inner_steps=1,cycles=1,
        learning_rate=.001,device="cpu",threads=2,precision="float64",image_precision="float32",
        loss="mind",strain_weight=.05,shape_weight=.0001,oob_weight=1.,method="radial",
        minimum_jacobian=.001,lr_calibration="edge",interpolation="p1_ac",output_selection="best_full")
    config=tmp_path/"configuration.json"
    config.write_text(json.dumps(dict(configuration=parameters)),encoding="utf-8")
    output=tmp_path/"new.npz";report=record(config,output)
    assert report["manual_labels_loaded_during_record"] is False and len(report["stages"])==4
    assert report["gradient_steps"]==4 and report["snapshot_cpu_bytes"]==4*1*9*9*2*8
    with np.load(tmp_path/"new_trajectory.npz") as data:
        assert data["vertices"].shape==(4,1,9,9,2)
        assert np.diff(data["optimizer_seconds"]).min()>0
    assert not (tmp_path/"old.npz").exists()
    with pytest.raises(FileExistsError):record(config,output)
