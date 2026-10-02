"""Preserve nine source RGB decodes from two existing decoders; no Torch or labels.

Seven historical and two current-decoder RGBs reconstruct their accepted512.
"""
import json
from pathlib import Path
import sys
import time
import subprocess
import PIL
from PIL import Image,features,ImageChops

ROOT=Path('D:/QC_optimization/.worktrees/phase6-dense-homeomorphism-plan')
DATA=Path('D:/QC_optimization_data/digital_topology_wsi')
DEST=DATA/'historical_rgb_decode_cache'
CURRENT_PYTHON=DATA/'dhr_clean_venv/Scripts/python.exe'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))

def main():
    if PIL.__version__!='10.3.0' or features.version('jpg')!='9.0':
        raise ValueError('the existing historical Pillow10.3/JPEG9 decoder is required')
    if (DEST/'manifest.json').exists():
        previous=read(DEST/'manifest.json')
        if len(previous['rows']) not in (7,9) or previous['decoder']['jpeg_version']!='9.0':
            raise ValueError('only extend or verify the existing known cache')
    sources=sorted({row[role] for row in read(ROOT/'outputs/coordinated_instance_registration/native22_inputs_t20/local_inputs.json')['rows'] for role in ('fixed','moving')})
    assert len(sources)==9 and len({Path(p).name for p in sources})==9
    accepted={}
    for name in ('cc10','cd31','ki67','prospc'):
        layout=read(DATA/'lung_lesion3_eval/canvas'/(name+'_layout.json'))
        accepted[Path(layout['moving']['source']).name]=layout['moving']
        if name=='cc10':accepted[Path(layout['fixed']['source']).name]=layout['fixed']
    for name in ('histo','rat_kidney'):
        for role in ('fixed','moving'):
            layout=read(DATA/'birl_anhir_dev/canvas'/(name+'_layout.json'))[role]
            accepted[Path(layout['source']).name]=layout
    assert set(accepted)=={Path(p).name for p in sources}
    (DEST/'rgb').mkdir(parents=True,exist_ok=True)
    manifest=dict(decoder=dict(python_executable=sys.executable,python_version=sys.version,
        pillow_version=PIL.__version__,jpeg_version=features.version('jpg'),
        operation='Image.open(original JPEG).convert(RGB), no resize, no EXIF transpose; lossless PNG'),
        annotations_read=False,rows=[],purpose='preserve decoded RGB for all nine sources across environments; seven historical default and two explicit current-decoder kidney overrides; no accepted canvas changed')
    current_decoder=json.loads(subprocess.check_output([str(CURRENT_PYTHON),'-c',
        'import json,sys,PIL;from PIL import features;print(json.dumps(dict(python_executable=sys.executable,python_version=sys.version,pillow_version=PIL.__version__,jpeg_version=features.version("jpg"),operation="Image.open(original JPEG).convert(RGB), no resize, no EXIF transpose; lossless PNG")))'],text=True))
    assert current_decoder['pillow_version']=='12.3.0' and current_decoder['jpeg_version']=='8.0'
    started=time.perf_counter()
    for value in sources:
        source=Path(value);layout=accepted[source.name];tick=time.perf_counter()
        with Image.open(source) as image:
            if image.getexif().get(274,1)!=1:raise ValueError('nonidentity original orientation')
            size=image.size
            rgb=image.convert('RGB') if not source.name.startswith('Rat-Kidney_') else None
        if rgb is None:
            raw=subprocess.check_output([str(CURRENT_PYTHON),'-c',
                'from PIL import Image;import sys;sys.stdout.buffer.write(Image.open(sys.argv[1]).convert("RGB").tobytes())',str(source)])
            rgb=Image.frombytes('RGB',size,raw)
        assert list(rgb.size)==layout['original_wh']
        canvas=Image.new('RGB',(512,512),'white')
        canvas.paste(rgb.resize(tuple(layout['resized_wh']),Image.Resampling.BILINEAR),tuple(layout['padding_xy']))
        with Image.open(layout['canvas_png']) as old:
            if ImageChops.difference(canvas,old.convert('RGB')).getbbox() is not None:
                raise ValueError('historical decode does not reproduce '+source.name)
        relative='rgb/'+source.name+'.png'
        if (DEST/relative).exists():
            with Image.open(DEST/relative) as saved:
                if saved.mode!='RGB' or saved.size!=rgb.size or ImageChops.difference(saved,rgb).getbbox() is not None:
                    raise ValueError('existing partial cache differs from original decode')
        else:rgb.save(DEST/relative,format='PNG',compress_level=1)
        manifest['rows'].append(dict(source=str(source),source_name=source.name,decoded_rgb=relative,
            original_wh=list(rgb.size),accepted512_reconstruction_exact=True,accepted512_canvas=layout['canvas_png'],
            accepted512_resized_wh=layout['resized_wh'],accepted512_padding_xy=layout['padding_xy'],
            decode_save_seconds=time.perf_counter()-tick))
        if source.name.startswith('Rat-Kidney_'):manifest['rows'][-1]['decoder']=current_decoder
        print(json.dumps(dict(source_name=source.name,original_wh=list(rgb.size),status='ok')),flush=True)
    manifest['elapsed_seconds']=time.perf_counter()-started
    (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(manifest=str(DEST/'manifest.json'),images=len(manifest['rows']),elapsed_seconds=manifest['elapsed_seconds'])),flush=True)

if __name__=='__main__':main()
