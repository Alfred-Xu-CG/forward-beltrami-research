"""Identify existing decoder that reconstructs each accepted RGB canvas exactly."""
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image
BASE=Path('D:/QC_optimization_data/digital_topology_wsi')
accepted={}
for category,names in [('lung_lesion3_eval',('cc10','cd31','ki67','prospc')),('birl_anhir_dev',('histo','rat_kidney'))]:
    for name in names:
        layout=json.loads((BASE/category/'canvas'/(name+'_layout.json')).read_text())
        for role in ('fixed','moving'):
            item=layout[role]
            if name!='cc10' and category=='lung_lesion3_eval' and role=='fixed':continue
            accepted[Path(item['source']).name]=item
child='from PIL import Image;import sys,json;x=json.loads(sys.argv[1]);im=Image.open(x["source"]).convert("RGB");c=Image.new("RGB",(512,512),"white");c.paste(im.resize(tuple(x["resized_wh"]),Image.Resampling.BILINEAR),tuple(x["padding_xy"]));sys.stdout.buffer.write(c.tobytes())'
for name,item in sorted(accepted.items()):
    expected=np.asarray(Image.open(item['canvas_png']).convert('RGB')).astype(int)
    result={}
    for label,python in [('historical','C:/Users/xuzhehao/anaconda3/python.exe'),('current','D:/QC_optimization_data/digital_topology_wsi/dhr_clean_venv/Scripts/python.exe')]:
        value=np.frombuffer(subprocess.check_output([python,'-c',child,json.dumps(item)]),dtype=np.uint8).reshape(512,512,3)
        difference=value.astype(int)-expected
        result[label]=dict(maximum=int(abs(difference).max()),different_components=int(np.count_nonzero(difference)))
    print(json.dumps(dict(source=name,**result)),flush=True)
