# Run using
# pip install snakemake-executor-plugin-slurm==0.4.1 snakemake==8.4.1
# snakemake --snakefile workflow/example.smk --workflow-profile workflow
# -e dryrun --dag | dot -Tpng > workflow/example.png

########################################

# This tells snakemake to check if the variables exist before attempting to run
envvars:
    "WANDB_API_KEY",

# Define important paths
project_name = "train_then_probe"
output_dir = "/srv/beegfs/scratch/groups/rodem/jetssl-lite/" + project_name + "/"
container: "~/scratch/images/jetssl-lite_master.sif" # Note the : this is for snakemake!
model_list = ["jepa", "jetgpt", "ssfm", "mpm"]

########################################

rule all:
    input:
        expand(f"{output_dir}{{m_name}}_ft/train_finished.txt", m_name=model_list)

rule finetune:
    output:
        f"{output_dir}{{m_name}}_ft/train_finished.txt"
    input:
        f"{output_dir}{{m_name}}/backbone.pkl"
    params:
        "scripts/train.py",
        "model=classifier",
        "network_name={m_name}_ft",
        "callbacks.backbone_finetune.unfreeze_at_step=9999999999",
        "datamodule.batch_size=500",
        "trainer.max_epochs=1",
        "full_resume=True",
        f"project_name={project_name}",
        f"model.backbone_path={output_dir}{{m_name}}/backbone.pkl",
    threads: 6
    resources:
        slurm_partition="shared-gpu,private-dpnc-gpu",
        runtime=12 * 60,  # minutes
        slurm_extra="--gres=gpu:1 --constraint=COMPUTE_TYPE_AMPERE",
        mem_mb=20000,
    wrapper:
        "file:hydra_cli"

rule pretrain:
    output:
        f"{output_dir}{{m_name}}/backbone.pkl"
    params:
        "scripts/train.py",
        "model={m_name}",
        "network_name={m_name}",
        "trainer.max_epochs=4",
        "full_resume=True",
        lambda w : f"datamodule={"jetclass_tokens" if w.m_name == "jetgpt" else "jetclass_masked"}",
    threads: 12
    resources:
        slurm_partition="shared-gpu,private-dpnc-gpu",
        runtime=12 * 60,  # minutes
        slurm_extra="--gres=gpu:1,VramPerGpu:20GB --constraint=COMPUTE_TYPE_AMPERE",
        mem_mb=20000,
    wrapper:
        "file:hydra_cli"
