"""Final benchmark for README numbers."""
import sys,os;sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','..'))
from cognicore.research.persistent_store import PersistentCognitionStore
from cognicore.nexus.run import run_nexus
from cognicore.nexus.trajectory_store import TrajectoryStore

PersistentCognitionStore().clear()
TrajectoryStore().clear()

for p in ['minimal','standard','test_first','review_first']:
    PersistentCognitionStore().clear()
    r = run_nexus(policy=p, max_attempts=5, quiet=True)
    att = sum(x['attempts'] for x in r['results'])
    tps = r['token_total'] // max(r['solved'], 1)
    fails = [x['task_id'] for x in r['results'] if not x['solved']]
    s = r['solved']
    t = r['total']
    tok = r['token_total']
    print(f"{p}|{s}/{t}|{att}|{tok}|{tps}|{fails}")
