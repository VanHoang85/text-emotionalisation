import json
import os

import datasets

_DESCRIPTION = """\
This new dataset is designed to for the task of style style transfer.
"""

GENERIC_DATA_DIR = os.path.join(os.path.join(os.getcwd(), "../data"), "generic")
CLASSIFIER_DATA_DIR = os.path.join(os.path.join(os.getcwd(), "../data"), "classifier")
EMOTION_DATA_DIR = os.path.join(os.path.join(os.getcwd(), "../data"), "style")


class StyleParaphrase(datasets.GeneratorBasedBuilder):
    """Style Paraphrase Dataset"""

    VERSION = datasets.Version("2.0.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="generic",
            version=VERSION,
            description="This dataset covers generic paraphrase data.",
        ),
        datasets.BuilderConfig(
            name="generic_nmt",
            version=VERSION,
            description="This dataset covers generic paraphrase data taken from NMT, Krishna filtered version.",
        ),
        datasets.BuilderConfig(
            name="classifier",
            version=VERSION,
            description="This dataset covers data for style classification.",
        ),
        datasets.BuilderConfig(
            name="neutralizer",
            version=VERSION,
            description="This dataset has emotion utterances as inputs, and neutral utts as targets",
        ),
        datasets.BuilderConfig(
            name="emo_stylizer",
            version=VERSION,
            description="This dataset has neutral utterances as inputs, and style utts as targets,"
                        "with style labels as constraints. No silver data added.",
        ),
        datasets.BuilderConfig(
            name="emo_stylizer_0.5",
            version=VERSION,
            description="This dataset has neutral utterances as inputs, and style utts as targets,"
                        "with style labels as constraints. Silver data equals half of gold data.",
        ),
        datasets.BuilderConfig(
            name="emo_stylizer_1.0",
            version=VERSION,
            description="This dataset has neutral utterances as inputs, and style utts as targets,"
                        "with style labels as constraints. Silver data equals gold data.",
        ),
        datasets.BuilderConfig(
            name="emo_stylizer_2.0",
            version=VERSION,
            description="This dataset has neutral utterances as inputs, and style utts as targets,"
                        "with style labels as constraints. Silver data doubles gold data.",
        ),
        datasets.BuilderConfig(
            name="phrase_stylizer",
            version=VERSION,
            description="This dataset has neutral utterances as inputs, and style utts as targets,"
                        "with style phrases as constraints.",
        ),
    ]

    DEFAULT_CONFIG_NAME = "generic"  # not mandatory to have a default configuration.

    def _info(self):
        # This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        if self.config.name == "classifier":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "utterance": datasets.Value("string"),
                    "label": datasets.Value("string")  # emo label
                }
            )
        elif self.config.name == "generic":
            features = datasets.Features(
                {
                    "reference": datasets.Value("string"),
                    "paraphrase": datasets.Value("string")
                }
            )
        elif self.config.name == "generic_nmt":
            features = datasets.Features(
                {
                    "reference": datasets.Value("string"),
                    "paraphrase": datasets.Value("string")
                }
            )
        elif self.config.name == "neutralizer":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "input": datasets.Value("string"),
                    "target": datasets.Value("string"),
                    "target_emo": datasets.Value("string"),
                    "context": datasets.Value("string"),  # previous utt
                    "context_emo": datasets.Value("string")
                }
            )
        else:
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "input": datasets.Value("string"),
                    "target": datasets.Value("string"),
                    "target_emo": datasets.Value("string"),
                    "constraint": datasets.Value("string"),  # can be either style or phrases
                    "context": datasets.Value("string"),  # previous utt
                    "context_emo": datasets.Value("string")
                }
            )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
        )

    def _split_generators(self, dl_manager):
        extension = 'txt' if "generic" in self.config.name else 'json'
        if "generic" in self.config.name:  # point to local dir
            data_dir = GENERIC_DATA_DIR
        elif self.config.name == "classifier":
            data_dir = CLASSIFIER_DATA_DIR
        else:
            data_dir = EMOTION_DATA_DIR

        if "emo_stylizer" in self.config.name and self.config.name.count("_") == 2:
            silver_per = self.config.name.split("_")[-1]
            if "2" not in silver_per:
                train_file = f"train_silver_{silver_per}.{extension}"
            else:
                train_file = f"train_{silver_per}.{extension}"
        else:
            train_file = f"train.{extension}"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, train_file),
                    "split": "train"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, f"test.{extension}"),
                    "split": "test"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, f"validation.{extension}"),
                    "split": "validation"
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        # TThis method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.

        if "generic" in self.config.name:
            with open(filepath, 'r', encoding="utf-8") as file:
                count = 0
                for line in file:
                    line = line.strip().split('\t')
                    if len(line) == 2:
                        count += 1
                        yield f"{split}_{count}", {  # Yields examples as (key, example) tuples
                            "reference": line[0],
                            "paraphrase": line[1],
                        }
        else:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)

            for id_, utt_info in data.items():
                if self.config.name == "classifier":
                    yield id_, {
                        "id": id_,
                        "utterance": utt_info["utterance"],
                        "label": utt_info["emo"]
                    }
                elif self.config.name == "neutralizer":
                    yield id_, {
                        "id": id_,
                        "input": utt_info["emotion_sent"],
                        "target": utt_info["neutral_sent"],
                        "target_emo": "neutral",
                        "context": utt_info["context"],
                        "context_emo": utt_info["context_emo"]
                    }
                elif "emo_stylizer" in self.config.name:
                    yield id_, {
                        "id": id_,
                        "input": utt_info["neutral_sent"],
                        "target": utt_info["emotion_sent"],
                        "target_emo": utt_info["emo"],
                        "constraint": utt_info["emo"],
                        "context": utt_info["context"],
                        "context_emo": utt_info["context_emo"]
                    }
                elif self.config.name == "phrase_stylizer":
                    yield id_, {
                        "id": id_,
                        "input": utt_info["neutral_sent"],
                        "target": utt_info["emotion_sent"],
                        "target_emo": utt_info["emo"],
                        "constraint": ". ".join(utt_info["phrases"]),  # maybe convert from list to string
                        "context": utt_info["context"],
                        "context_emo": utt_info["context_emo"]
                    }
