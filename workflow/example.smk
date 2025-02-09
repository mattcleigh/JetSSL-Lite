# Run using
# pip install snakemake-executor-plugin-slurm==0.4.1 snakemake==8.4.1
# snakemake --snakefile workflow/example.smk --workflow-profile workflow
# -e dryrun --dag | dot -Tpng > workflow/example.png

########################################

# This tells snakemake to check if the variables exist before attempting to run
envvars:
    "WANDB_API_KEY"

# Required for running on apptainer
container:
    "/home/users/l/leighm/scratch/images/jetssl-lite_master.sif"

# Define important paths
project_name = "train_then_probe"
output_dir = "/srv/beegfs/scratch/groups/rodem/jetssl-lite"
model_list = ["jepa", "ssfm", "mpm", "gpt"]

########################################

wildcard_constraints:
    m_name = "[0-9a-zA-Z]+" # Makes sure the model_name can't have underscores

rule all:
    input:
        expand(f"{output_dir}/{project_name}/{{m_name}}_ft/done.txt", m_name=model_list)

rule finetune:
    output:
        f"{output_dir}/{project_name}/{{m_name}}_ft/done.txt"
    input:
        f"{output_dir}/{project_name}/{{m_name}}/done.txt"
    params:
        "scripts/train.py",
        "model=classifier",
        "network_name={m_name}_ft",
        "callbacks=finetune",
        "callbacks.backbone_finetune.unfreeze_at_step=-1",
        "datamodule.batch_size=1000",
        "full_resume=True", # Will autostart new if checkpoint is missing
        "trainer.max_epochs=1",
        f"project_name={project_name}",
        f"model.backbone_path={output_dir}/{project_name}/{{m_name}}/backbone.pkl",
    threads: 6
    resources:
        slurm_partition="shared-gpu,private-dpnc-gpu",
        runtime=12 * 60,  # minutes
        slurm_extra="--gres=gpu:1,VramPerGpu:20GB --constraint=COMPUTE_TYPE_AMPERE",
        mem_mb=20000,
    wrapper:
        "file:hydra_cli"

rule pretrain:
    output:
        f"{output_dir}/{project_name}/{{m_name}}/done.txt"
    params:
        "scripts/train.py",
        "model={m_name}",
        "network_name={m_name}",
        "full_resume=True", # Will autostart new if checkpoint is missing
        "trainer.max_epochs=5",
        f"project_name={project_name}",
        lambda w : f"datamodule={'jetclass_tokens' if w.m_name == 'jetgpt' else 'jetclass_masked'}",
        lambda w : "datamodule.batch_size=250" if w.m_name == "jetgpt" else "",
    threads: 12
    resources:
        slurm_partition="shared-gpu,private-dpnc-gpu",
        runtime=24 * 60 * 4,  # minutes
        slurm_extra="--gres=gpu:1,VramPerGpu:20GB --constraint=COMPUTE_TYPE_AMPERE",
        mem_mb=20000,
    wrapper:
        "file:hydra_cli"
