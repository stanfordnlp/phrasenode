# Mapping natural language commands to web elements

## Data

Due to its large size, the data is hosted outside Github:
<https://nlp.stanford.edu/projects/phrasenode/>

You can download the dataset by running the script as follows.

```
bash download_dataset.sh
```

## Setup

The code was developed in the following environment:

- Python 2.7
- Pytorch 0.4.1
- CUDA 9

To install dependencies:

- (Optional) Create a virtualenv / conda environment
  ```
  virtualenv.py -p python2.7 env
  source env/bin/activate
  ```

- Python dependencies
  ```
  sudo apt-get install python-dev
  pip install -r requirements.txt
  ```

Alternatively, use the docker image [`ppasupat/phrasenode`](https://hub.docker.com/r/ppasupat/phrasenode/)
For latest image: `docker pull ppasupat/phrasenode:1.06`

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

## Configurations

Here are the configerations used in the final experiments:

* `base.txt`: Used in all experiments
* `models/encoding.txt`: The embedding-based method
* `models/alignment.txt`: The alignment-based method
* `node-embedder/allan.txt`: The node embedder as described in the paper
* `ablation/*.txt`: Ablation

Note that the visual neighbor is off by default.
To turn it on, use `general/neighbors.txt`.

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

## Demo Chrome extension

* Start the server with
  ```
  export WEBREP_DATA=./data
  ./server.py data/experiments/0_testrun/config.txt -m data/experiments/0_testrun/checkpoints/20000.checkpoint/model
  ```
  where `0_testrun` should be changed to the model's directory, and `20000` should be changed to the checkpoint number you want.

* Install the unpacked Chrome extension in `demo/phrasenode-demo`
  * Follow the instruction from [here](https://developer.chrome.com/extensions/getstarted#manifest) to load the extension. (The manifest file should already be in `demo/phrasenode-demo`)
  * An extension button should now show up on the toolbar
  * On any web page, click on the extension button and enter a phrase in the prompt that pops up.
  * If the server does not throw an error, the selected element should be highlighted with a red border. Details can be viewed in the developer console.

## Referenece

> Panupong Pasupat, Tian-Shun Jiang, Evan Liu, Kelvin Guu, Percy Liang.  
> [**Mapping natural language commands to web elements.**](https://arxiv.org/abs/1808.09132)  
> Conference on Empirical Methods in Natural Language Processing (EMNLP), 2018.

**CodaLab:** <https://worksheets.codalab.org/worksheets/0x0097f249cd944284a81af331093c3579/>
