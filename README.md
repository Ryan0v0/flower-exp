# client_sampling

<!-- :warning: **Note this repo makes use of ClientProxy.get_properties() which is not yet available in a Flower release. For this to work you need to install the current main branch of the Flower repo** :warning: -->

## Setting up your working environment

This create a Conda environment ready for this project:
```bash
# fresh conda environment
conda create -n client_sampling python=3.8 -y
# access environment
/home/wz341/anaconda3/bin/conda activate client_sampling
# get a lasting PyTorch version
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia -y
# install requirements
pip install -r requirements.txt
```

## Define your own sampling strategy

> If you want to completely disable the sampling, use `class: null` for the `sampler` in the config.

A generic [`Sampler`](src/sampling/sampler.py) class contins just two methods:  `get_client_fn()`, which returns a function to be run on the client; and, `decide()` which runs on the server and processes all the results returned from each client running the function returned by `get_client_fn()`. This is how it looks like:

```python
class Sampler(ABC):

    @staticmethod
    def get_client_fn():
        """This should return the function that the clients will execute and return to result server."""
        pass

    @staticmethod
    def decide(clients_info: List[Tuple[ClientProxy, PropertiesRes]]) -> List[ClientProxy]:
        """The Server executes this function by processing the results sent by
        the clients. Then this function returns a list of clients proxies to sample
        for normal FL training. If desired, an upper limit on how many clients
        to sample in the round can be used."""
        pass
```

Then, you can implement your own sampler by specifying what to run in each of those methods. For example, if we want a Sampler that takes the Top-N clients (out of M) based on a value these return, such Sampler will look like this:

```python
class TopKRandomSampler(Sampler):
    """Clients are asked to return a random scalar, then the strategy will select
    the N clients that returned the highest values among the M clients queried. A
    more meaningful extension of this sampler would ask clients to evaluate their
    local validation set, then select the N clients with lowest validation loss."""
    def __init__(self, sample_size: int, clients_per_round: int):
        super().__init__()
        self.sample_size = sample_size  # M
        self.clients_per_round = clients_per_round  # N

    @staticmethod
    def get_client_fn():
        def client_fn(args: Dict):
            """Returns a random integer between [0, 1e6]."""
            import random
            return {'random_val': random.randint(0, 1e6)}

        return client_fn

    def decide(self, clients_info: List[Tuple[ClientProxy, PropertiesRes]]) -> List[ClientProxy]:
        """Selects the Top-N clients according to the returned integer."""

        # sort list according to returned value in PropertiesRes structure
        sorted_info = sorted(clients_info, key = lambda x: x[1].properties['random_val'], reverse = True)

        # now return a list of ClientProxy objects for .fit()
        # here we take the top-N
        return [s_client_info[0] for s_client_info in sorted_info[:self.clients_per_round]]
```

## Run a single config

```bash
python main.py --config_file configs/simple.yaml
```
Use bash:
```bash
#!/bin/bash
#SBATCH --cpus-per-task 8 # cpu resources
#SBATCH --gres=gpu:a40:1 # gpu resources
#SBATCH --job-name=wanru-Client_Sampling # a name just for you to identify your job easily

# source your conda environment (which should live in Aoraki)
source /nfs-share/wz341/miniconda3/bin/activate client_sampling

srun python main.py --config_file configs/simple.yaml # the `srun` is not strictly needed but is nice to use since you'll be able to do `sattach <job_id>.0` to check the progress of your job.
```
* Then you’d submit the script like this: `sbatch <my_script_name>.sh` . After you run that, you’ll get a number, that’s your job id
* Do `sattach <jobid>.0`  to see the progress (e.g. you’ll see in this way all the print statements in your experiments.
* If you do `squeue` you’ll see the SLURM job queue. You can see there the jobs that are running and pending

## Run a sweep of configs

```bash
wandb sweep --entity camlsys --name <A MEANINGFUL NAME FOR THIS EXP> configs/simple_sweep.yaml
```

This will generate sweep project in W&B and output some text. You can ignore the warnings. For example:

```bash
(pytorch) ➜  client_sampling git:(main) ✗ wandb sweep --entity camlsys configs/simple_sweep.yaml --name dummy_v2
wandb: Creating sweep from: configs/simple_sweep.yaml
wandb: WARNING Malformed sweep config detected! This may cause your sweep to behave in unexpected ways.
wandb: WARNING To avoid this, please fix the sweep config schema violations below:
wandb: WARNING   Violation 1. Additional properties are not allowed ('strategy', 'dataset', 'conda_env', 'augment', 'rounds', 'cpus', 'val_ration', 'arch', 'wandb_entity', 'groupnorm', 'lr_decay', 'dataset_fl_root', 'optimizer', 'pool_size', 'vram', 'global_eval_every_n_rounds', 'path_to_data', 'end_lr' were unexpected)
wandb: Created sweep with ID: a24ck2nv
wandb: View sweep at: https://wandb.ai/camlsys/client_sampling/sweeps/a24ck2nv
wandb: Run sweep agent with: wandb agent camlsys/client_sampling/a24ck2nv
```

Then if you run as said above (just execute `wandb agent camlsys/client_sampling/a24ck2nv` in a terminal), each experiment in the sweep will run, one after another.

**However**, we can do better and parallelize this with SLURM (so multiple runs in the sweep run in parallel). To achieve this you'll need to create (or edit) a script like this one:

```bash
#!/bin/bash
#SBATCH --cpus-per-task 7
#SBATCH --gres=gpu:1 # ONE GPU PER JOB SHOULD BE ENOUGH
#SBATCH --job-name=<GIVE IT SOME RANDOM NAME>
#SBATCH --array=0-3%2 # THIS DEFINES AN ARRAY OF 4 JOBS ALLOWING TO RUN AT MOST 2 IN PARALLEL. YOU NEED TO UPDATE THE `3` WITH THE TOTAL NUMBER OF RUNS IN YOUR SWEEP
#SBATCH --exclude=tarawera 

# source your conda environment (which should live in Aoraki)
source /nfs-share/${USER}/miniconda3/bin/activate flower

# just copy the same command as before but this time tell it to just run one job (the next in the queue)
srun wandb agent --count 1 camlsys/client_sampling/a24ck2nv
```

Make this script executable with `chmod +x <your_script>.sh` and then run it `sbatch <your_script>.sh`.
