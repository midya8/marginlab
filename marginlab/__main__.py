"""Strict offline JSON CLI."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from .experiment import run,saved,audit
from .core import Model
from .controls import controls

def read_json(file):
    def constant(value):raise ValueError('nonfinite JSON value')
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('duplicate JSON key: '+key)
            result[key]=value
        return result
    return json.loads(Path(file).read_text(),parse_constant=constant,object_pairs_hook=pairs)

def main():
    p=argparse.ArgumentParser(description='Inspectable soft-margin classification, offline, not clinical validation');sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('report');sub.add_parser('controls');a=sub.add_parser('fit');a.add_argument('--output',required=True);a.add_argument('--budget',type=int,default=50000)
    a=sub.add_parser('audit');a.add_argument('--report');a=sub.add_parser('predict');a.add_argument('--input',required=True);a.add_argument('--report');a.add_argument('--model-id')
    a=sub.add_parser('certificate');a.add_argument('--model-id');a.add_argument('--report');a.add_argument('--output',required=True)
    a=sub.add_parser('verify-certificate');a.add_argument('--input',required=True)
    args=p.parse_args()
    try:
        if args.command=='certificate':
            from .certificate import produce,verify
            r=read_json(args.report) if args.report else saved();id=args.model_id or r['selected_id'];item=next((a for a in r['candidates'] if a['id']==id),None)
            if item is None:raise ValueError('unknown model ID')
            proof=produce(item['model']);Path(args.output).write_text(json.dumps(proof,separators=(',',':'),allow_nan=False)+'\n');out=dict(output=args.output,model_id=id,**verify(proof))
        elif args.command=='verify-certificate':
            from .verify_certificate import check
            out=check(read_json(args.input))
        elif args.command=='report':
            r=saved();out={k:r[k] for k in ('selected_id','protocol','baselines','scope')}
        elif args.command=='controls':out=controls()
        elif args.command=='fit':
            r=run(max_updates=args.budget);Path(args.output).write_text(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n');out=dict(output=args.output,selected_id=r['selected_id'],statuses={a['id']:a['status'] for a in r['candidates']})
        elif args.command=='audit':out=audit(read_json(args.report) if args.report else saved())
        else:
            r=read_json(args.report) if args.report else saved();id=args.model_id or r['selected_id'];item=next((a for a in r['candidates'] if a['id']==id),None)
            if item is None:raise ValueError('unknown model ID')
            value=read_json(args.input)
            if not isinstance(value,dict) or set(value)!={'samples'} or not isinstance(value['samples'],list) or any(not isinstance(row,list) or any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in row) for row in value['samples']):raise ValueError('input must contain only samples, arrays of JSON numbers')
            x=np.array(value['samples'],dtype=float)
            if x.ndim!=2 or x.shape[1]!=30 or not len(x) or not np.isfinite(x).all():raise ValueError('expected nonempty rows of 30 finite original feature values')
            model=Model.from_dict(item['model']);score=model.decision((x-r['mean'])/r['scale']);out=dict(model_id=id,scores=score.tolist(),predictions=np.where(score>=0,1,-1).tolist(),scope='signed decisions on historical labels, not probabilities, clinical validation or medical advice')
        print(json.dumps(out,indent=2,allow_nan=False));return 0
    except (ValueError,KeyError,TypeError,ArithmeticError,OSError,np.linalg.LinAlgError) as error:
        print('marginlab: '+str(error),file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
