# Text Emotionalisation

### Abstract

While emotion generation in dialogues for empathetic systems has received special attention from the research community, emotion transfer in texts is a relatively new task. This thesis aims to explore the methods to emotionalise the texts while ensuring fluency and preserving the original semantic meaning. 
Instead of using unsupervised methods, together with a data-driven approach to the problem of _style_ and _content_ as it is normally pursued in literature, we attempt to differentiate the two terms. Our effort, thus, leads to a parallel neutral-emotional corpus. 
Two Transformer-based sequence-to-sequence architectures are adopted for the implementation of our text emotionalisation models. An additional emotion style loss is employed to guide the generation towards more emotional words and phrases. Before fine-tuning the pre-trained sequence-to-sequence models into the emotionalisation models, we first train them on paraphrase data to refine their re-writing capacity and thus improve the preservation of original content in the generated candidates.
The encouraging results of our initial experiments suggest the potential of our approach. 
Despite having a small-scale corpus, the models are able to emotionalise the input text. The ablation studies are further conducted to understand the contribution of two architecture designs, namely the emotion style loss during training and the pre-training paraphrasing stage.
However, both automatic and human results show that their contribution is modest and unclear. We believe a more comprehensive evaluation is needed to investigate this issue further. 

_A Master Thesis project done at Stuttgart Technology Center, Sony Europe B.V._


## Prepare the virtual environment 
- The codes run on Python 3.8+.
- To install dependencies needed for the models, in your virtual environment, run ``pip install -r requirements.txt``


## How to train the models
Note: To understand the arguments in the ``.sh`` files, 
please see the corresponding ``.py`` files.

### How to train the emotion classifier

Run the ``run_classifier.sh`` file, which runs the ``style_classifier.py`` file.
Check the .sh file for training arguments.


### How to train the paraphrase (BART) models

Note: For PEGASUS, we make use of the pre-trained paraphrase model at ``https://huggingface.co/tuner007/pegasus_paraphrase``.

Run the ``run_paraphraser.sh`` file, which runs the ``paraphraser.py`` file.
We use BERTScore for evaluating the model, and thus the training take more than a day. 
To reduce training time, one can (1) change the evaluating method, (2) evaluating on fewer samples.


### How to train the emotionalisation models

Run either ``run_bart.sh`` or ``run_pegasus.sh``. They both, in turn, run the ``style_paraphraser.py`` file.
Check the .py file for explanations on the arguments. Some to note:

- ``--use_style_loss``: whether to use emotion style loss during training. In this case, make sure you have the correct path to the emotion classifier.
- ``--use_bertscore``: whether to use bertscore during evaluation (will increase training time)
- ``--dataset_config_name``: the dataset to use for the training, which is defined in the ``style_dataset.py`` file. ``emo_stylizer`` is for training emotionalisation models while ``neutralizer`` is for the neutralisation models.


## How to run and generate texts using the emotionalisation models
### Full generation process (with filtering and ranking)
If you want to run the entire generation process from the beginning to the end as described in the thesis,
you should use the ``run_generation.sh`` file, which runs the ``generation.py`` file.

Two most important things to pay attention to:
1. Specify which model to use for generation.
2. What is the input/output file

In the ``run_generation.sh`` file, you also need to specify the Slurm arguments.

For the models' arguments, 

- To specify the model, modify ``--model_name_or_path``. Use whichever model you have trained.
- To sepcify the type of model, use ``--dataset_config_name emo_stylizer``.
- To specify the output filename, use ``--output_file`` argument. The default output path is set to ``./outputs``,
which you can change with the argument ``--output_dir``.
- To specify the input file, one can
    - run the model on **the test split** of the dataset, using the argument ``--dataset_name "./style_dataset.py"``
    - run the model on **an external input file**, using the argument ``--input_file "path_to_external_input_file"``. However, you need to pay attention to the format of the file. Check the method ``load_data_from_file()`` from ``generation.py``. A sample file can be found at ``"data/style/new_test.json"``.

- To ask the models to generate all possible emotions, use argument ``--gen_all_emos``

### Only generation process (no filtering and ranking) on Shell 

```
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, PegasusConfig

model_name = "path_to_your_emotionaliser"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = PegasusConfig.from_pretrained(model_name)
tokenizer = PegasusTokenizer.from_pretrained(model_name, use_fast=True)
model = PegasusForConditionalGeneration.from_pretrained(model_name, config=config).to(device)

def emotionalise(input_text, target_emotion, num_return_sequences, num_beams):
  batch = tokenizer([input_text], target_emotion, padding='longest', truncation=True, max_length=60, return_tensors="pt").to(device)
  embs = model.generate(**batch, max_length=60, num_beams=num_beams, num_return_sequences=num_return_sequences)
  output_text = tokenizer.batch_decode(embs, skip_special_tokens=True)
  return output_text 
```


