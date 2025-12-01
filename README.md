# Project Summary
This project compares several different transformer architectures on image classification tasks. These architectures include:
* basic vision transformer and basic cnn as baselines
* Recursive transformer model with state and full backpropagation through all recursive applications
* Recursive transformer model with only ``stage-wise'' backpropagation, HRM-style
* Recursive transformer model with HRM-style backprop (as above) but with injected timestep embedding

# Technical Details
This project uses `uv` for Python project management. It uses the standard in configuration and logging (hydra/omegaconf, wandb). 