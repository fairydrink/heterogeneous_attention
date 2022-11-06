import logging
import os

from datasets.arrow_dataset import _concatenate_map_style_datasets
from pandas import DataFrame
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput

cur_file = os.path.realpath(__file__)

import re
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import precision_score
from datasets import Dataset


import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    MBartTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed, AutoModelForQuestionAnswering, SquadDataset, squad_convert_examples_to_features,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding, BertModel, AutoModel, DataCollatorForTokenClassification
)

from transformers import PegasusTokenizer, PegasusForConditionalGeneration, PegasusConfig, set_seed, Trainer, \
    default_data_collator, DataCollatorForLanguageModeling, MBartTokenizer, Trainer, TrainingArguments

from transformers.trainer_utils import is_main_process
from datasets import load_dataset, load_metric
from dataclasses import dataclass, field
from typing import Optional

from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:

    task: str = field(
        default="qa",
        metadata={
            "help": "The name of the task, should be qa (or qa_{dataset} for evaluating "
                    "pegasus)"
        },
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    context_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for qa)."},
    )
    question_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for qa)."},
    )
    answer_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for qa)."},
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                    "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                    "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                    "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
                    "value if set."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if not self.task.startswith("classify"):
            raise ValueError(
                "`task` should be qa or qa_{dataset}"
            )
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


@dataclass
class DataValidationArguments:
    min_summ_length: Optional[int] = field(
        default=100,
        metadata={
            "help": "The minimum length of the sequence to be generated."
        },
    )
    max_summ_length: Optional[int] = field(
        default=300,
        metadata={
            "help": "The maximum length of the sequence to be generated."
        },
    )

    num_beams: Optional[int] = field(
        default=3,
        metadata={
            "help": "Number of beams for beam search. 1 means no beam search."
        },
    )
    length_penalty: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer sequences."
        },
    )

    no_repeat_ngram_size: Optional[int] = field(
        default=2,
        metadata={
            "help": " If set to int > 0, all ngrams of that size can only occur once."
        },
    )


name_mapping = {
    "imdb": ("text", "label")
}


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, DataValidationArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, test_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, test_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)

    if data_args.dataset_name is not None:
        datasets = []
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["test"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files)

    class CustomClassifier(nn.Module):
        def __init__(self, num_class):
            super(CustomClassifier, self).__init__()
            self.model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
            self.num_class = num_class
            self.dropout = nn.Dropout(0.25)
            self.layer1 = nn.Linear(768, 1024)
            self.layer2 = nn.Linear(1024, 512)
            self.layer3 = nn.Linear(512, 256)
            self.classifier1 = nn.Linear(256, num_class)

        def forward(self, input_ids, attention_mask, token_type_ids,labels):
            outputs = self.model(
                input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            )
            embeddings = outputs[1]
            embeddings = self.dropout(embeddings)
            embeddings = self.layer1(embeddings)
            embeddings = self.layer2(embeddings)
            embeddings = self.layer3(embeddings)
            logits = self.classifier1(embeddings)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_class), labels.view(-1))
            return  SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # if training_args.do_train:
    #     column_names = datasets["train"].column_names
    # else:
    #     column_names = datasets["test"].column_names

    # text_column, label_column = None, None
    #
    # if data_args.task.startswith("classify"):
    #     dataset_columns = name_mapping.get(data_args.dataset_name, None)
    #     if data_args.context_column is None:
    #         text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    #     else:
    #         text_column = data_args.context_column
    #     if data_args.question_column is None:
    #         label_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    #     else:
    #         label_column = data_args.question_column
    label_map = {'1_-3':0,'0_-1':1,'2_-2':2,'1_-2':3,'0_-2':4,'0_-3':5}

    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False
    global n
    n = 0

    def preprocess_function(examples):
        texts = examples["text"]
        labels = examples["label"]
        source_id = examples["source_id"]
        inputs = [source_id[i] + " " + inp for i,inp in enumerate(texts)]

        model_inputs = tokenizer(inputs, max_length=128,padding=True)
        model_inputs["labels"] = [label_map[str(item) + "_" +source_id[i]] for i,item in enumerate(labels)]
        return model_inputs


    datasets1 = load_dataset("emotion")
    train_dataset1 = datasets1["train"]
    if data_args.max_train_samples is not None:
        train_dataset1 = train_dataset1.select(range(3000))
        train_dataset1 = train_dataset1.add_column(name="source_id",column=['-1']*3000)
    train_dataset1 = train_dataset1.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=["text", "label","source_id"],
        load_from_cache_file=not data_args.overwrite_cache,
    )

    datasets2 = load_dataset("financial_phrasebank",'sentences_allagree')
    train_dataset2 = datasets2["train"]
    if data_args.max_train_samples is not None:
        train_dataset2 = train_dataset2.select(range(2000))
        train_dataset2 = train_dataset2.rename_column("sentence", "text")
        train_dataset2 = train_dataset2.add_column("source_id", np.array(['-2'] * 2000))
    train_dataset2 = train_dataset2.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=["text", "label","source_id"],
        load_from_cache_file=not data_args.overwrite_cache,
    )

    datasets3 = load_dataset("rotten_tomatoes")
    train_dataset3 = datasets3["train"]
    if data_args.max_train_samples is not None:
        train_dataset3 = train_dataset3.select(range(3000))
        train_dataset3 = train_dataset3.add_column("source_id", np.array(['-3'] * 3000))
    train_dataset3 = train_dataset3.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=["text", "label","source_id"],
        load_from_cache_file=not data_args.overwrite_cache,
    )

    train_dataset = _concatenate_map_style_datasets([train_dataset3,train_dataset2,train_dataset1])

    num_class = len(set(train_dataset['labels']))
    model = CustomClassifier(num_class)

    if training_args.do_eval:
        eval_dataset1 = datasets1["train"]
        if data_args.max_train_samples is not None:
            eval_dataset1 = eval_dataset1.select(range(3000,3300))
            eval_dataset1 = eval_dataset1.add_column(name="source_id", column=['-1'] * 300)
        eval_dataset1 = eval_dataset1.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=["text", "label", "source_id"],
            load_from_cache_file=not data_args.overwrite_cache,
        )

        eval_dataset2 = datasets2["train"]
        if data_args.max_train_samples is not None:
            eval_dataset2 = eval_dataset2.select(range(2000,2264))
            eval_dataset2 = eval_dataset2.rename_column("sentence", "text")
            eval_dataset2 = eval_dataset2.add_column("source_id", np.array(['-2'] * 264))
        eval_dataset2 = eval_dataset2.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=["text", "label", "source_id"],
            load_from_cache_file=not data_args.overwrite_cache,
        )

        eval_dataset3 = datasets3["train"]
        if data_args.max_train_samples is not None:
            eval_dataset3 = eval_dataset3.select(range(3000,3300))
            eval_dataset3 = eval_dataset3.add_column("source_id", np.array(['-3'] * 300))
        eval_dataset3 = eval_dataset3.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=["text", "label", "source_id"],
            load_from_cache_file=not data_args.overwrite_cache,
        )

        eval_dataset = _concatenate_map_style_datasets([eval_dataset3, eval_dataset2, eval_dataset1])

    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Metric
    metric_name = "precision"
    metric = load_metric(metric_name)


    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds = np.argmax(preds,axis=1)
        result = metric.compute(predictions=preds, references=labels)
        return result

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Training
    if training_args.do_train:

        train_result = trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    results = {}
    if training_args.do_eval:

        model = trainer.model
        tokenizer = trainer.tokenizer
        print("\n")
        print("Running Evaluation Script")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        output = []
        inco = 0

        min_length = test_args.min_summ_length
        max_length = test_args.max_summ_length

        df_test = eval_dataset
                
        result = trainer.predict(eval_dataset)
        
        output_eval_file = os.path.join(training_args.output_dir, "evaluation_scores.txt")


        with open(output_eval_file, "w") as writer:
                logger.info("***** eval results *****")
                for key, value in sorted(result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")


        
        print("Evaluation scores saved in {}".format(output_eval_file))

    return results


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()
