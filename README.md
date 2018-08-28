# Mapping natural language commands to web elements

## Setup

- Python dependencies
  ```
  pip install requirements.txt
  ```

- PyTorch: Install by following directions from <http://pytorch.org/>.

- Data (UNDER CONSTRUCTION)

## Quick start

If you just want to see something happen:

```
export WEBREP_DATA=./data
./main.py configs/base.txt configs/model/encoding.txt configs/node-embedder/allan.txt -n testrun
```

- This executes the main entrypoint script `main.py` with three config files.
- The config files are specified in HOCON format. If multiple files are specified,
  they will be merged together (with later configs overwriting values from previous ones).
- You can also add ad-hoc config strings using the `-c` option. These are applied last.
- `-n` specifies the experiment directory name.

## Experiment management

All training runs are managed by the `PhraseNodeTrainingRuns` object. For example,
to get training run #141, do this:

```python
runs = PhraseNodeTrainingRuns()   # Note the final "s"
run = runs[141]  # a PhraseNodeTrainingRun object
```

A `TrainingRun` is responsible for constructing a model, training it, saving it
and reloading it (see superclasses `gtd.ml.TrainingRun` and
`gtd.ml.TorchTrainingRun` for details.)

The most important methods on `PhraseNodeTrainingRun` are:
- `__init__`: the model, data storage, etc, are initialized
- `train`: actual training of the model happens here

## TensorBoard

Statistics are logged to TensorBoard. To view:

```
tensorboard --logdir=data/experiments
```
