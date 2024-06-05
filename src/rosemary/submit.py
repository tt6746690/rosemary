import re
import os
import subprocess
import shlex


shell_scripts_template_slurm = """
echo "Running on $SLURM_JOB_NODELIST"
echo "======"

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
master_port=10002
RDZV_ENDPOINT=$master_addr:$master_port

source ~/.profile
conda activate {conda_env}
cd {cwd}

set -e
set -x
echo "======"

srun {cmd}

[ ! -f "{log_dir}/$SLURM_JOB_ID*.out" ] && mv {log_dir}/$SLURM_JOB_ID*.out {save_dir} ||:
"""



def multiline_to_singleline(cmd):
    cmd = cmd.strip()
    cmd = re.sub(r'\\(?![$])', '', cmd) # replace all '\' but not '\$'
    cmd = cmd.replace('\n', '')
    cmd = re.sub(' +', ' ', cmd)
    cmd = cmd.strip()
    return cmd


def submit_job_slurm(
    shell_scripts: str,
    job_name='wpq-job',
    partition='learnai4p',
    nodes=1,
    num_cpus=1,
    cpu_mem=3,
    num_gpus=0,
    log_path=None,
    test_run=False,
    num_jobs=1,
    shell_scripts_modification_fn=None,
):
    """
        submit to SLURM scheduler (via `sbatch`) a job that executes `shell_scripts`,
            with specified resources.

        Usage
        ```
        from rosemary.submit import submit_job_slurm

        # submit simple bash commands, with minimal resources
        submit_job_slurm('echo hello world', partition='learnai4p')

        # test job chaining
        out = submit_job_slurm('echo foo; echo bar', partition='learnai4p', test_run=False, num_jobs=2, job_name='test.out', shell_scripts_modification_fn=lambda x: x.replace('bar', 'baz'))
        ```
    """
    if isinstance(shell_scripts, list):
        return [
            submit_job_slurm(
                shell_scripts=shell_scripts,
                job_name=job_name,
                partition=partition,
                nodes=nodes,
                num_cpus=num_cpus,
                cpu_mem=cpu_mem,
                num_gpus=num_gpus,
                log_path=log_path,
                test_run=test_run,
                num_jobs=num_jobs,
                shell_scripts_modification_fn=shell_scripts_modification_fn) 
            for x in shell_scripts]

    if log_path is None:
        log_path = os.path.join(os.getcwd(), '%J.out')
    if '.out' not in log_path:
        raise ValueError('log_path must contain ".out"')
    if log_path is None:
        log_path = os.path.join(os.getcwd(), '%J.out')

    log_dir = os.path.dirname(log_path)
    log_filename = os.path.basename(log_path)
    log_filename_noext, ext = log_filename.split('.')

    info = []
    job_id = '<job_id>'
    for i in range(num_jobs):
        # for >1 jobs, the `log_filename` is modified with `_i:num_jobs` before extension.
        if num_jobs > 1:
            log_path = os.path.join(
                log_dir, '.'.join([log_filename_noext+f'_{i+1}:{num_jobs}', ext]))
            

        if shell_scripts_modification_fn is not None:
            shell_scripts_cmd = shell_scripts_modification_fn(shell_scripts) \
                if i != 0 else shell_scripts
        else:
            shell_scripts_cmd = shell_scripts
        shell_scripts_cmd = shell_scripts_cmd.strip().split('\n')
        shell_scripts_cmd = [x for x in shell_scripts_cmd if x != '']
        shell_scripts_cmd = '; '.join(shell_scripts_cmd)
        # escape double quotes in `shell_scripts_cmd` to wrap it in a double quote.
        shell_scripts_cmd = shell_scripts_cmd.replace('"', '\\"') 
        shell_scripts_cmd = '"' + shell_scripts_cmd + '"'

        sbatch_cmd = f"""
        sbatch \
            --job-name={job_name} \
            --partition={partition} \
            --nodes={nodes} \
            --cpus-per-task={num_cpus} \
            --mem={cpu_mem}gb \
            --gres=gpu:{num_gpus} \
            --output={log_path} \
            --wrap={shell_scripts_cmd} \
        """

        sbatch_cmd = multiline_to_singleline(sbatch_cmd)
        sbatch_cmd = shlex.split(sbatch_cmd)

        if i != 0:
            sbatch_cmd.extend(["--dependency", f"afterok:{str(job_id)}"])
            
        if test_run:
            job_info = {'args': ' '.join(sbatch_cmd),}
        else:
            try:
                p = subprocess.Popen(sbatch_cmd,
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE)
                stdout, stderr = p.communicate()
                stdout = stdout.decode("utf-8")
                match = re.search(r"Submitted batch job (\d+)", stdout)
                job_id = int(match.group(1)) if match else stdout
            except Exception as e:
                pass
            job_info = {'args': ' '.join(sbatch_cmd), 'job_id': job_id}
        info.append(job_info)
    
    return info

