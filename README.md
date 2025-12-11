# Project Summary
This project compares several different transformer architectures on image classification tasks. These architectures include:
* basic vision transformer and basic cnn as baselines
* Recursive transformer model with state and full backpropagation through all recursive applications
* Recursive transformer model with only ``stage-wise'' backpropagation, HRM-style
* Recursive transformer model with HRM-style backprop (as above) but with injected timestep embedding

# Usage
This project uses `uv` for Python project management. Ensure `uv` Python package manager is installed. Then run:
```
uv sync
```
This will install all dependencies. Then set up the venv:
```
source .venv/bin/activate
```
 

In addition, several global configuraton variables are in `.env`. Create one if it doesn't exist with the following fields:
```
DATASET_DIR= # dir to save and load datasets from. Useful to avoid duplicates if you're using these datasets elsewhere.
WANDB_PROJECT_NAME= # ditto
```

(Optional) Add this to your `.git/config`:
```
[filter "strip-notebook-output"]
clean = "jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR"
```

This cleans jupyter notebooks of metadata on commit.

(Optional) Login to wandb for logging.
```
wandb login
```

To start training a model, run
```
python main.py
```

Parameters can be overriden using hydra/omegaconf, e.g.
```
python main.py arch.config.max_recursion_steps=6
```

This project supports distributed training. Ex. 

```
torchrun --standalone --nproc_per_node=auto main.py dataset=cifar100 arch=full_recursive_vit arch.config.mlp_dim=128 arch.config.dim_head=16
```