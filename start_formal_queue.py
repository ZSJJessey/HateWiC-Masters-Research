"""Launch one authenticated local execution supervisor; no paid or remote service."""
from pathlib import Path
import sys,os,json,fcntl,subprocess
import psutil
ROOT=Path(__file__).resolve().parent

def launch():
    runtime=ROOT/'.runtime';runtime.mkdir(exist_ok=True)
    with (runtime/'queue_start.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        pidfile=runtime/'formal_queue.pid'
        if pidfile.exists():
            try:
                pid=int(pidfile.read_text());process=psutil.Process(pid)
                if process.is_running() and str(ROOT/'execute_formal.py') in process.cmdline():
                    return {'status':'already_running','supervisor_pid':pid,'progress':str(ROOT/'run_progress.json')}
            except (ValueError,psutil.Error):pass
        identity=json.loads((ROOT/'verification/formal_execution_identity.json').read_text())
        import hashlib
        if hashlib.sha256((ROOT/'local_runner.py').read_bytes()).hexdigest()!=identity['runner_sha256']:
            raise RuntimeError('Frozen local runner source changed; inspect before launch.')
        with (runtime/'formal_queue.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,str(ROOT/'execute_formal.py')],cwd=ROOT,stdout=log,stderr=log,start_new_session=True)
        pidfile.write_text(str(proc.pid))
        # Temporary idle-sleep inhibition belongs only to this training supervisor.
        # It does not change persistent macOS power settings or override lid closure.
        caffeine=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(proc.pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        (runtime/'caffeinate.pid').write_text(str(caffeine.pid))
        return {'status':'launched','supervisor_pid':proc.pid,'progress':str(ROOT/'run_progress.json'),'scope':'140 first-pass fits, then 320 core repeats, then verified reports; originals preserved'}

if __name__=='__main__':print(json.dumps(launch()))
