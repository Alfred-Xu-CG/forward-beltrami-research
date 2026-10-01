"""Coordinator's separate Fraction-area recomputation of six saved audit fixtures."""
import argparse
import json
from fractions import Fraction
from pathlib import Path


def check(path,output):
    if output.exists():raise FileExistsError(output)
    report=json.loads(path.read_text(encoding="utf-8"));rows={}
    for name,row in report["results"].items():
        fixture=row["minimum_fixture"]
        points=[]
        for label in fixture["corner_triangle_labels"]:
            node=fixture["nodes"][label]
            point=tuple(Fraction.from_float(float.fromhex(v)) for v in node["mapped_unit_float_hex"])
            reference=((Fraction(node["column"])+Fraction(1,2))/512,
                       (Fraction(node["row"])+Fraction(1,2))/512)
            reconstructed=tuple(reference[i]+Fraction.from_float(node["saved_pixel_displacement"][i])/512 for i in range(2))
            if point!=reconstructed:raise AssertionError("stored minimum fixture frame reconstruction disagrees")
            points.append(point)
        # Homogeneous 3x3 determinant, independently of difference cross-product.
        a,b,c=points
        determinant=a[0]*(b[1]-c[1])-a[1]*(b[0]-c[0])+b[0]*c[1]-b[1]*c[0]
        normalized=determinant*512**2
        if normalized>=0:raise AssertionError("reported minimum not exactly negative")
        if abs(float(normalized)-fixture["normalized_determinant"])>1e-12:
            raise AssertionError("minimum value disagrees with separate exact-area path")
        rows[name]=dict(exact_fraction_numerator=normalized.numerator,
            exact_fraction_denominator=normalized.denominator,normalized_value=float(normalized),
            pixel_frame_reconstruction_exact=True,negative=True)
    result=dict(source=str(path),method="separate Fraction homogeneous-area formula plus pixel-frame reconstruction",results=rows)
    output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();print(json.dumps(check(args.report,args.output)))
