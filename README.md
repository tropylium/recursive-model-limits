# Project Summary
This project compares several different transformer architectures on image classification tasks. These architectures include:
* basic vision transformer and basic cnn as baselines
* Recursive transformer model with state and full backpropagation through all recursive applications
* Recursive transformer model with only ``stage-wise'' backpropagation, HRM-style
* Recursive transformer model with HRM-style backprop (as above) but with injected timestep embedding

# Technical Details
This project uses `uv` for Python project management. It uses the standard in configuration and logging (hydra/omegaconf, wandb). 

In addition, several global configuraton variables are in `.env`. Create one if it doesn't exist with the following fields:
```
DATASET_DIR= # dir to save and load datasets from. Useful to avoid duplicates if you're using these datasets elsewhere.
WANDB_PROJECT_NAME= # ditto
```

In addition, add this to your `.git/config`:
```
[filter "strip-notebook-output"]
clean = "jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR"
```

This cleans jupyter notebooks of metadata on commit.