import logging
import os
import sys
import time
import random
from copy import deepcopy
import collections
import datasets
from datasets import load_from_disk
import numpy as np
import torch
import transformers
import evaluate
from datasets import load_dataset, DatasetDict
from transformers import AutoConfig, AutoTokenizer, EvalPrediction, default_data_collator, DataCollatorWithPadding
from transformers import (HfArgumentParser, TrainingArguments, PretrainedConfig,
                          glue_output_modes, glue_tasks_num_labels, set_seed)
from transformers.models.bert.modeling_bert import (
    BertAttention, BertEmbeddings, BertEncoder, BertForQuestionAnswering,
    BertForSequenceClassification, BertLayer, BertModel, BertOutput,
    BertSelfAttention, BertSelfOutput, QuestionAnsweringModelOutput)
from args import AdditionalArguments, DataTrainingArguments
from utils.cofi_utils import *
from models.l0_module import L0Module
from models.modeling_bert import CoFiBertForSequenceClassification
from models.modeling_roberta import CoFiRobertaForSequenceClassification
from trainer.trainer import CoFiTrainer 
from utils.utils import *
from models.model_args import ModelArguments
import train_utils
#import wandb
import data.snli as snli
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "snli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, AdditionalArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, additional_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, additional_args = parser.parse_args_into_dataclasses()
    training_args.dataloader_num_workers=4
    #print("training args ", training_args)
    os.makedirs(training_args.output_dir, exist_ok=True)
    
     # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # save args
    torch.save(data_args, os.path.join(
        training_args.output_dir, "data_args.bin"))
    torch.save(model_args, os.path.join(
        training_args.output_dir, "model_args.bin"))
    torch.save(additional_args, os.path.join(
        training_args.output_dir, "additional_args.bin"))

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # print all arguments
    log_all_parameters(logger, model_args, data_args,
                       training_args, additional_args)

    
    max_data = 1000 if data_args.data_debug > 0 else None
    train,val, _,_ = train_utils.create_dataloaders(max_data)
    label_list = list(set(train.labels))
    # Labels
    is_regression=False
    if is_regression:
        num_labels = 1
    else:
        num_labels = len(set(train.labels))
    print("BEFORE LOADING MODEL THE NUM OF LABELS IS ", num_labels)
    t_name=data_args.task_name
    config = AutoConfig.from_pretrained(
        'bert-base-uncased',
        num_labels=num_labels,
        finetuning_task=t_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        'bert-base-uncased',
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # set up configuration for distillation
    if additional_args.do_distill:
        config.output_attentions = True
        config.output_hidden_states = True
        
 
    Model = CoFiBertForSequenceClassification if model_args.model_name_or_path.startswith(
        "bert") else CoFiRobertaForSequenceClassification
    print("loading teacher")
    teacher_model = None
    if additional_args.do_distill:
        
        teacher_model = Model.from_pretrained(
            pretrained_model_name_or_path=additional_args.distillation_path, #BERT/model_best.pth
            train_data=train,
            max_data=max_data,
            teacher=True,
            config=config,
        )
        teacher_model.eval() #! inside has a cofibertmodel #! CofiBertForSequenceClassification

    config.do_layer_distill = additional_args.do_layer_distill #! True
    
    model = Model.from_pretrained(
        pretrained_model_name_or_path=None, #Pretrained model
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        teacher=False,
        config=config,
        train_data=train,
        max_data=max_data,
        output_dir = training_args.output_dir,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    ) #! inside the function, we get the original struct  #! CofiBertForSequenceClassification
    
    model.config.label2id = {f"LABEL_{i}":i for i in range(num_labels)}
    # initialize the layer transformation matrix to be an identity matrix
    if additional_args.do_layer_distill:
        initialize_layer_transformation(model)

    logger.info(model)
    logger.info(f"Model size: {calculate_parameters(model)}")

    zs = None
    
    if additional_args.pretrained_pruned_model is not None:
        zs = load_zs(additional_args.pretrained_pruned_model)
        model = load_model(additional_args.pretrained_pruned_model, Model, zs)
        print(
            f"Model Size after pruning: {calculate_parameters(model)}")

    l0_module = None
    if additional_args.pruning_type is not None:
        l0_module = L0Module(config=config,
                             droprate_init=additional_args.droprate_init,
                             temperature=additional_args.temperature,
                             target_sparsity=additional_args.target_sparsity,
                             pruning_type=additional_args.pruning_type)

    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None

    LABEL_STOI = {"entailment": 0, "neutral": 1, "contradiction": 2}
    LABEL_ITOS = {v: k for k, v in LABEL_STOI.items()}
    model.config.id2label = LABEL_ITOS
    model.config.label2id = LABEL_STOI
    label_to_id = LABEL_STOI
    
    print("===Mapping =====", model.config.label2id)
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    
    print("Line 312 UPDATED PROCESS INPUTS")
    
    vocab= {'stoi': train.stoi, 'itos': train.itos}
    def preprocess_function(examples):
        result = {}
        def indices_to_bert_tokens(indices):
            batch_size, seq_len = indices.shape
      
            words = []
            for i in range(batch_size):
                sentence = []
                for idx in indices[i]:
                    if idx.item() in vocab['itos']:
                        word = vocab['itos'][idx.item()]
                        if word not in ("[PAD]", "<pad>", "PAD"): 
                            sentence.append(word)
                    else:
                        break
                words.append(sentence)

            return tokenizer(words, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
        
        s1_pad, s1len, s2_pad, s2len, labels = snli.pad_collate([
            (torch.tensor(p), plen, torch.tensor(h), hlen, l)
            for p, plen, h, hlen, l in zip(
                examples["premise"],
                examples["premise_len"],
                examples["hypothesis"],
                examples["hypothesis_len"],
                examples["label"]
            )
        ])

        # Transpose as needed
        s1_transposed = s1_pad.transpose(1, 0)
        s2_transposed = s2_pad.transpose(1, 0)

        # Convert to BERT tokens
        s1_tokens = indices_to_bert_tokens(s1_transposed)
        s2_tokens = indices_to_bert_tokens(s2_transposed)

        # Build the result with proper batch dimension
        result = {
            "pre_input_ids": s1_tokens["input_ids"].cpu().numpy(),
            "pre_attention_mask": s1_tokens["attention_mask"].cpu().numpy(),
            "hyp_input_ids": s2_tokens["input_ids"].cpu().numpy(),
            "hyp_attention_mask": s2_tokens["attention_mask"].cpu().numpy(),
            "label": [l if l != -1 else -1 for l in labels]
        }
            
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        hf_train = Dataset.from_dict({
            "premise": train.s1s,  # Replace with your actual attributes
            "premise_len": train.s1lens,  # Replace with your actual attributes
            "hypothesis": train.s2s,
            "hypothesis_len": train.s2lens,
            "label": train.labels
        })
        train = hf_train.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            remove_columns=["premise", "premise_len", "hypothesis", "hypothesis_len"],
            desc="Running tokenizer on dataset",
        ) #! get dataset
        train = train.filter(
            lambda example: example["label"] != -1,
            desc="Filtering out samples with label -1"
        )
        
        train.save_to_disk("proceesed_train")
        reloaded_train = load_from_disk("proceesed_train")
        assert train.features == reloaded_train.features
        
        hf_val = Dataset.from_dict({
            "premise": val.s1s,  # Replace with your actual attributes
            "premise_len": val.s1lens,  # Replace with your actual attributes
            "hypothesis": val.s2s,
            "hypothesis_len": val.s2lens,
            "label": val.labels
        })
        val = hf_val.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            remove_columns=["premise", "premise_len", "hypothesis", "hypothesis_len"],
            desc="Running tokenizer on dataset",
        ) #! get dataset
        val = val.filter(
            lambda example: example["label"] != -1,
            desc="Filtering out samples with label -1"
        )
        val.save_to_disk("proceesed_val")
        reloaded_val = load_from_disk("proceesed_val")
        assert val.features == reloaded_val.features

    # Get the metric function
    metric = evaluate.load("accuracy")
    import gc
    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
     
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        labels = p.label_ids
        if hasattr(preds, 'device'):  # If it's a torch tensor
            preds = preds.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

        gc.collect()
        torch.cuda.empty_cache()
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=labels)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - labels) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == labels).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    logger.info(
        f"************* {len(train)} Training Examples Loaded *************")
    logger.info(
        f"************* {len(val)} Evaluation Examples Loaded *************")

    #model.load_state_dict(load_file("/workspace/CoFiPruning/out/MNLI/CoFi/MNLI_sparsity0.95/model.safetensors"))
    trainer = CoFiTrainer(
        model=model,
        args=training_args,
        additional_args=additional_args,
        train_dataset=train if training_args.do_train else None,
        eval_dataset=val if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        l0_module=l0_module,
        teacher_model=teacher_model,
  
    )

    if training_args.do_train:
        trainer.train()
        
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)
        print(trainer.evaluate())

    


if __name__ == "__main__":
    # wandb.init(project='Cofi')
    os.environ["WANDB_DISABLED"] = "true"
    t_start = time.time()
    main()
    t_end = time.time()
    logger.info(f"Training took {round(t_end - t_start, 2)} seconds.")



   

