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
model_list = ["jepa", "ssfm", "mpm", "dino", "gpt"]

########################################

wildcard_constraints:
    m_name = "[0-9a-zA-Z]+" # Makes sure the model name can't have underscores

rule all:
    input:
        expand(f"{output_dir}/{project_name}/{{m_name}}_ft/done.txt", m_name=model_list)

rule finetune:
    output:
        f"{output_dir}/{project_name}/{{m_name}}_ft/done.txt"
    input:
        f"{output_dir}/{project_name}/{{m_name}}/done.txt"
    resources:
        slurm_extra = "--gres=gpu:1,VramPerGpu:20GB --constraint=COMPUTE_TYPE_AMPERE",
    shell:
        f"""
        python scripts/train.py \
        model=classifier \
        network_name={{wildcards.m_name}}_ft \
        callbacks=finetune \
        callbacks.backbone_finetune.unfreeze_at_step=-1 \
        datamodule.batch_size=1000 \
        trainer.max_epochs=1 \
        project_name={project_name} \
        model.backbone_path={output_dir}/{project_name}/{{wildcards.m_name}}/backbone.pkl \
        """

rule pretrain:
    output:
        f"{output_dir}/{project_name}/{{m_name}}/done.txt"
    params:
        datamodule = lambda w : "jetclass_tokens" if w.m_name == "gpt" else "jetclass_masked",
        batch_size = lambda w : 250 if w.m_name == "gpt" else 1000,
        max_epochs = lambda w : 1 if w.m_name == "gpt" else 2,
    resources:
        runtime=3 * 24 * 60,  # minutes
        slurm_extra = "--gres=gpu:1,VramPerGpu:20GB --constraint=COMPUTE_TYPE_AMPERE",
    shell:
        f"""
        python scripts/train.py \
        model={{wildcards.m_name}} \
        network_name={{wildcards.m_name}} \
        trainer.max_epochs={{params.max_epochs}} \
        project_name={project_name} \
        datamodule={{params.datamodule}} \
        datamodule.batch_size={{params.batch_size}} \
        """

