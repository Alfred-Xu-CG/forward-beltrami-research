"""Independent non-GPU lifecycle checks of the four-case transfer wrapper."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json

from tools import coordinated_fusion_dispatch_benchmark as b


def main():
    for failure in (None, 'first', 'compiled', 'probe', 'later'):
        events = []
        with TemporaryDirectory(prefix='independent_dispatch_') as td:
            output = Path(td)/'result'
            def load(_):
                return [dict(name=n, anchor='anchor', configuration=dict(
                    fixed='f', moving='m', affine='a', matches='p',
                    match_weight=.2, unchanged=['full', 'fusion'])) for n in b.CASES]
            def execute(cfg, plan):
                manifest = json.loads((output/'benchmark.json').read_text())
                assert sum(len(c['runs']) for c in manifest['cases']) == 56
                assert manifest['annotations_read'] is False
                case = cfg.output.parent.name
                events.append((case, plan['backend']))
                assert cfg.unchanged == ['full', 'fusion'] and cfg.match_weight == .2
                assert cfg.match_p1_sampling == ('existing' if plan['backend']=='eager' else 'frozen')
                if (failure=='first' and len(events)==1 or failure=='compiled' and len(events)==2
                    or failure=='later' and len(events)==5):
                    raise RuntimeError('independently injected failure')
                return dict(**plan, output=str(cfg.output), status='complete')
            def probe(cfg, anchor, *, outputpath):
                case=outputpath.parent.name
                assert events[-2:] == [(case, 'eager'),(case,'inductor')]
                events.append((case,'probe'))
                return dict(passed=failure!='probe',status='complete')
            def comparisons(rows):
                return [dict(affine_boundary_interpolation_equal=True,counters_equal=True)]
            with patch.object(b.comparison,'compare_runs',comparisons), \
                 patch.object(b.comparison,'numerical_summary',lambda _: {}), \
                 patch.object(b.comparison,'warm_summary',lambda _: {}):
                result=b.run('unused',output,executor=execute,probe=probe,case_loader=load)
            rows=[r for c in result['cases'] for r in c['runs']]
            assert len(rows)==56 and result['all_attempts_terminal']
            assert all(r['status']!='pending' for r in rows)
            assert result['speed_evidence_eligible'] == (failure is None)
            expected_attempts={None:56,'first':1,'compiled':2,'probe':2,'later':4}[failure]
            assert result['attempted_calls']==expected_attempts
            if failure is None:
                expected=[arm for _ in range(3) for arm in ['eager','inductor','inductor','eager']]
                for case in b.CASES:
                    assert [arm for name,arm in events if name==case] == ['eager','inductor','probe',*expected]
            before=(output/'benchmark.json').read_bytes()
            try:
                b.run('unused',output,case_loader=load)
            except FileExistsError:
                pass
            else:
                raise AssertionError('existing output was accepted')
            assert (output/'benchmark.json').read_bytes()==before
        print(f'{failure or "all_success"}: PASS ({expected_attempts}/56 attempts)')


if __name__=='__main__':
    main()
