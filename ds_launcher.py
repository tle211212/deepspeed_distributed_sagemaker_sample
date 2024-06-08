import os
import subprocess
import json
import logging
from argparse import ArgumentParser
import shutil
import stat
import time

logger = logging.getLogger(__name__)

# upgrade flash attention here
#try:
#    os.system("MAX_JOBS=4 pip install flash-attn --no-build-isolation --upgrade")
#except:
#    print("flash-attn failed to install")


def parse_args():
    parser = ArgumentParser(
        description=("SageMaker DeepSpeed Launch helper utility that will spawn deepspeed training scripts")
    )
    # positional
    parser.add_argument(
        "--training_script",
        type=str,
        help="Path to the training program/script to be run in parallel, can be either absolute or relative",
    )

    # rest from the training program
    parsed, nargs = parser.parse_known_args()

    return parsed.training_script, nargs


def gen_hostfile(node_count, gpu_count):
    s = '\n'.join([f"algo-{i} slots={gpu_count}" for i in range(1, node_count+1)])
    with open('./deepspeed_hostfile', 'w') as f:
        f.write(s)
    print('Deepspeed hostfile content:')
    print(s)


def setup_ssh_passwordless():
    with open('/root/.ssh/authorized_keys', 'a') as authkey_file:
        with open('configs/id_rsa.pub', 'r') as pubkey_file:
            shutil.copyfileobj(pubkey_file, authkey_file)
    shutil.copy('configs/id_rsa', '/root/.ssh/')
    shutil.copy('configs/id_rsa.pub', '/root/.ssh/')
    os.chmod('/root/.ssh/id_rsa', stat.S_IREAD)


def run_ssh_command(nodeAddr, cmd):
    sshCmd = f'ssh {nodeAddr} {cmd}'
    return os.system(sshCmd)


# Check if all worker nodes have the ready flag set
def check_all_workers_ready(workerAddrList):
    for w in workerAddrList:
        if run_ssh_command(w, 'cat /tmp/jj_ready') != 0:
            print(f'Node {w} is not ready')
            return False
    print('All worker nodes are ready')
    return True


# After master node is done, it set the flag on all worker nodes
# so that the worker nodes can terminate themselves
def set_workers_done(workerAddrList):
    for w in workerAddrList:
        print(f'Set Done flag on {w}')
        run_ssh_command(w, 'touch /tmp/jj_done')
        run_ssh_command(w, 'cat /tmp/jj_done')


def main():
    # https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/launcher/launch.py
    num_gpus = int(os.environ.get("SM_NUM_GPUS", 0))
    hosts = json.loads(os.environ.get("SM_HOSTS", "{}"))
    num_nodes = len(hosts)
    current_host = os.environ.get("SM_CURRENT_HOST", 0)
    rank = hosts.index(current_host)
    print(f"num_gpus = {num_gpus}, num_nodes = {num_nodes}, current_host = {current_host}, rank = {rank}")

    # os.environ['NCCL_DEBUG'] = 'INFO'

    # get number of GPU
    if num_gpus == 0:
        raise ValueError("No GPUs found.")

    # Deepspeed distributed training requires the following setup
    # 1. gen hostfile
    gen_hostfile(num_nodes, num_gpus)

    # gen ssh key pair and setup sshd to enable passwordless ssh login among all nodes.
    # 2. Passwordless ssh login among all nodes
    setup_ssh_passwordless()
    os.system('/usr/sbin/sshd')

    # 3. Tell Deepspeed to pass the LD_LIBRARY_PATH which contains Conda stuff when 
    #    it executes works on remote nodes
    os.system('echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH > ./.deepspeed_env')
    os.system('echo PATH=$PATH >> ./.deepspeed_env')
    os.system('cat ./.deepspeed_env')
    os.system('chmod a+rwx /tmp')
    os.system('chmod o+t /tmp')
    # On main node, Deepspeed use pdsh to send works to remote nodes
    os.system('/usr/bin/apt-get update')
    os.system('/usr/bin/apt-get install -y pdsh libaio1 libaio-dev')

    if rank == 0:
        train_script, args = parse_args()
        command = f"deepspeed --hostfile=./deepspeed_hostfile {train_script} {' '.join(args)}"
        print(f'{current_host} is waiting for all worker nodes to become ready')
        while (not check_all_workers_ready(hosts[1:])) and (len(hosts) > 1):
            time.sleep(10)
            print(f'{current_host} is waiting for all worker nodes to become ready')

        # launch deepspeed training
        print(f"Launch deepspeed on first node command = {command}")
        try:
            deepspeed_launch(command)
        except:
            pass
        print('Training FINISHED')
        # Tell worker nodes to shutdown
        if (len(hosts) > 1):
            set_workers_done(hosts[1:])
    else:
        print(f'Node {current_host} is waiting for node 0 to distribute work')
        os.system('touch /tmp/jj_ready')
        while (os.system('cat /tmp/jj_done') != 0):
            print(f'Node {current_host} is working, check done again in 30s')
            time.sleep(30)


def deepspeed_launch(command):
    # try:
    try:
        subprocess.run(command, shell=True)
    except Exception as e:
        logger.info(e)


if __name__ == "__main__":
    main()