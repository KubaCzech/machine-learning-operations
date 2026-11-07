## HW 1: Pytorch, Pytorch Lightning, Model monitoring, Hyper-parameter optimization

The goal of this homework is to familiarize yourself with Pytorch Lightning, which serves as a tool for organizing all things related to training (this is a semi-quote), and to perform hyper-parameter optimization (including model monitoring with a model monitoring tool of your choice).

When it comes to Pytorch Lightning, this is the one time you are required to use this particular package, with no option to choose another, as I feel there are no real alternatives (at least for Pytorch). Of course, I'm open to being proven wrong.

### Example flow to complete the homework:
* Prepare a Dataset class and a Dataloader class using Pytorch for an arbitrary dataset of your choice (see the dataset guidelines below).
* Prepare a training script for any chosen model in Pytorch Lightning. To do this, you need to create two modules: a Data Module and a Lightning Module.
* Add model monitoring using the built-in Pytorch Lightning callback/logger for a chosen monitoring tool, for instance Wandb or MLFlow.
* Validate model learning: confirm that the model does in fact learn by comparing the training and validation loss curves.
* Implement optimization of any chosen parameter (or group of parameters) using a chosen optimization package (I recommend Optuna or RayTune).

### Choosing a dataset:
* You can choose an arbitrary dataset. I encourage you to use datasets from your other projects, e.g. your engineering thesis, Master's thesis, or other ongoing research.
* **Do not choose a problem that is too simple**, i.e. one where current architectures already reach 100% on the metrics (e.g. MNIST-level tasks). In that case hyper-parameter optimization is pointless, as there is nothing left to improve.
* Choose a problem that is not yet solved, i.e. one without 100% metrics on every benchmark, so that there is real room for improvement.

### Additional organization information:
* If possible, please don't copy whole solutions outright, like the one linked below.
* Please upload your code to a repository.
* Feel free to use models from any of your previous or future (e.g. Master's degree) projects.

### Acceptance Criteria:
* The code is uploaded to a repository and contains a Data Module and a Lightning Module.
* Training is logged in a chosen monitoring tool, with both training and validation curves visible.
* The model achieves reasonable values of the chosen metrics on the validation set, and the training/validation curves show that it actually learns.
* **Choose appropriate metrics for your task.** For example, if your dataset has class imbalance, use F1-score (or a similar metric) instead of plain accuracy.
* Hyper-parameter optimization yields an improvement in validation metrics of **at least 3%** compared to the baseline configuration.
* If you fail to obtain this improvement, that is also acceptable, **but you have to justify it (to me in person, do not waste time writing any reports etc) and show your work**: what was optimized, which search space was used, what the results of the trials were, and why, in your opinion, no improvement was achieved.

### Presentation Guidelines:
The acceptance criteria above apply to the whole group. Additionally, a small subset of students can present their results in front of the whole group.

5 minute presentation (no slides necessary) showcasing:
* Purpose of your model (machine learning task, data type).
* Lightning Module.
* Results of training in the chosen monitoring tool, with training and validation curves. Describe at least one change you made to the parameters / your training setup based on the training and validation curves. This can be as small as a learning rate adjustment, but we of course welcome different ones. The goal is to try to draw insights from the logged hyper-parameters, curves, and metrics.
* Results of hyper-parameter optimization: show which parameters you decided to optimize and how this improved the validation metrics.
* Main problems encountered (for others to learn from your mistakes).

#### Tryhard section (optional, for presenters who want to go above&beyond, not obligatory to obtain points, just aura points):
Pick any of the challenges below:
* **Your worst run:** show the run or trial that went most wrong (diverging loss, heavy overfitting, a pruned Optuna trial) and explain why it happened.
* **Hyper-parameter importance:** show which parameter mattered most (e.g. Optuna's `plot_param_importances` or the Wandb parameter importance panel, or Shap values or any other feature attribution) and say whether it matched your intuition beforehand.
* **Predict before you reveal:** before showing the optimization results, ask the group to guess which configuration won (or what the best learning rate was).
* **One-sentence takeaway:** finish with a single concrete lesson that others could apply to their own projects.

### Grading:
* The homework yields **1 point**.
* The presentation yields **1 additional point** (2 points in total for students who present).

### Deadline:
* Classes on 09.11.2026
* Submitting the homework late yields a one-time penalty of 0.25 points.

### Useful links:
* https://lightning.ai/lightning-ai/studios/image-segmentation-with-pytorch-lightning
* https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html
* https://freedium.cfd/https://pub.aimind.so/optimizing-hyperparameters-with-optuna-a-hands-on-tutorial-bb8e2ffd1801
* https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_lightning_simple.py
* https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html
