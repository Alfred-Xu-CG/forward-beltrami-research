"""Read-only decoded-RGB versus resize boundary probe across installed runtimes."""
import json
from pathlib import Path
import subprocess
import numpy as np
import PIL
from PIL import Image,features

base=Path('D:/QC_optimization_data/digital_topology_wsi/lung_lesion3_eval/canvas')
layout=json.loads((base/'cc10_layout.json').read_text())
child='from PIL import Image;import sys;sys.stdout.buffer.write(Image.open(sys.argv[1]).convert("RGB").tobytes())'
result=[]
for role in ('fixed','moving'):
    item=layout[role];source=item['source'];wh=item['original_wh']
    old=subprocess.check_output(['C:/Users/xuzhehao/anaconda3/python.exe','-c',child,source])
    old=np.frombuffer(old,dtype=np.uint8).reshape(wh[1],wh[0],3)
    current=np.asarray(Image.open(source).convert('RGB'))
    expected=np.asarray(Image.open(base/('cc10_'+role+'512.png')).convert('RGB'))
    def error(first,second):
        difference=first.astype(int)-second.astype(int)
        return dict(maximum=int(np.abs(difference).max()),different_components=int(np.count_nonzero(difference)))
    def render(array):
        canvas=Image.new('RGB',(512,512),'white')
        canvas.paste(Image.fromarray(array).resize(tuple(item['resized_wh']),Image.Resampling.BILINEAR),tuple(item['padding_xy']))
        return np.asarray(canvas)
    result.append(dict(role=role,original_rgb_current_vs_anaconda=error(current,old),
        current_resize_current_decode_vs_accepted=error(render(current),expected),
        current_resize_anaconda_decode_vs_accepted=error(render(old),expected)))
print(json.dumps(dict(current_pillow=PIL.__version__,current_jpeg=features.version('jpg'),rows=result),indent=2))
