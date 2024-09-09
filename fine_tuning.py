import torch
import pandas as pd
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset, Audio, DatasetDict
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
import argparse
import re


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["WANDB_PROJECT"] = "Speech"  # name your W&B project
# os.environ["WANDB_CACHE_DIR"] = "wandb/.cache"
# os.environ["HF_HOME"] = "/cs/usr/tomer.cohen13/lab/speach/Project/huggingface/cache"
# os.environ["HF_HUB_CACHE"] = "/cs/usr/tomer.cohen13/lab/speach/Project/huggingface/cache"
# HUG_CACHE = "/cs/usr/tomer.cohen13/lab/speach/Project/huggingface/cache"


# for local run we use cache_dir=HUG_CACHE in evaluate.load("wer") and evaluate.load("cer")
WER_METRIC = evaluate.load("wer")
CER_METRIC = evaluate.load("cer")


def clean_string(string):
    return re.sub(r'[^A-Za-z0-9 ]+', '', string.lower()).strip()


def prepare_dataset(batch, feature_extractor_, tokenizer_):
    # load and resample audio data from 48 to 16kHz
    audio = batch["Wav_path"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor_(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer_(batch["Prompts"]).input_ids
    return batch


# put together a list of samples into a mini training batch, https://www.youtube.com/watch?v=-RPeakdlHYo
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def compute_metrics_for_evaluation(pred_ids, label_ids, tokenizer_):

    # replace -100 with the pad_token_id
    for i, mat in enumerate(label_ids):
        label_ids[i][mat == -100] = tokenizer_.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = [clean_string(s) for s in tokenizer_.batch_decode(pred_ids, skip_special_tokens=True)]
    label_str = [clean_string(s) for s in tokenizer_.batch_decode(label_ids, skip_special_tokens=True)]

    wer_col = [100 * WER_METRIC.compute(predictions=[p], references=[r]) for p, r in zip(pred_str, label_str)]
    cer_col = [100 * CER_METRIC.compute(predictions=[p], references=[r]) for p, r in zip(pred_str, label_str)]

    return {"wer": wer_col, "cer": cer_col, "pred_str": pred_str, "label_str": label_str, "wer_all": 100 * WER_METRIC.compute(predictions=pred_str, references=label_str), "cer_all": 100 * CER_METRIC.compute(predictions=pred_str, references=label_str)}


def compute_metrics(pred, tokenizer_):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer_.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = [clean_string(s) for s in tokenizer_.batch_decode(pred_ids, skip_special_tokens=True)]
    label_str = [clean_string(s) for s in tokenizer_.batch_decode(label_ids, skip_special_tokens=True)]

    wer = 100 * WER_METRIC.compute(predictions=pred_str, references=label_str)
    cer = 100 * CER_METRIC.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='which Whisper model to use', default="whisper-small", choices=["whisper-tiny", "whisper-base", "whisper-small", "whisper-medium", "whisper-large-v3"])
    parser.add_argument('-e', '--epochs', help='Number of training epochs', default=5000, type=int)
    parser.add_argument('-b', '--batch_size', help='Batch size', default=16, type=int)
    parser.add_argument('-l', '--learning_rate', help='Learning rate', default=1e-5, type=float)
    parser.add_argument('-w', '--weight_decay', help='Weight Decay for AdamW', default=0, type=float)
    parser.add_argument('-r', '--warmup_ratio', help='Warmup ratio for the learning rate', default=0.1, type=float)
    parser.add_argument('-d', '--dropout', help='The dropout probability for the training', default=0.0, type=float)
    parser.add_argument('-s', '--seed', help='The random seed for training', default=42, type=int)
    parser.add_argument('-n', '--run_name', help='Weights and biases run name', required=True)

    args = parser.parse_args()

    data_info_train = pd.read_csv('archive/Dysarthria and Non Dysarthria/Torgo/processed_data_train.csv')
    data_info_val = pd.read_csv('archive/Dysarthria and Non Dysarthria/Torgo/processed_data_val.csv')
    data_info_test = pd.read_csv('archive/Dysarthria and Non Dysarthria/Torgo/processed_data_test.csv')

    train_dataset = Dataset.from_pandas(data_info_train).cast_column("Wav_path", Audio(sampling_rate=16000))
    val_dataset = Dataset.from_pandas(data_info_val).cast_column("Wav_path", Audio(sampling_rate=16000))
    test_dataset = Dataset.from_pandas(data_info_test).cast_column("Wav_path", Audio(sampling_rate=16000))

    common_voice = DatasetDict()
    common_voice["train"] = train_dataset
    common_voice["val"] = val_dataset
    common_voice["test"] = test_dataset

    # for local run we use cache_dir=HUG_CACHE in WhisperFeatureExtractor.from_pretrained() and WhisperTokenizer.from_pretrained()
    feature_extractor = WhisperFeatureExtractor.from_pretrained(f"openai/{args.model}", language="English")
    tokenizer = WhisperTokenizer.from_pretrained(f"openai/{args.model}", language="English", task="transcribe",)

    common_voice = common_voice.map(lambda x: prepare_dataset(x, feature_extractor, tokenizer), num_proc=1, remove_columns=common_voice.column_names["train"])
    # for local run we use cache_dir=HUG_CACHE in WhisperProcessor.from_pretrained()
    processor = WhisperProcessor.from_pretrained(f"openai/{args.model}", language="English", task="transcribe")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # for local run we use cache_dir=HUG_CACHE in WhisperForConditionalGeneration.from_pretrained()
    model = WhisperForConditionalGeneration.from_pretrained(f"openai/{args.model}")

    # model.config.suppress_tokens = []
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"
    model.config.forced_decoder_ids = None
    model.config.dropout = args.dropout

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"trained_models/{args.run_name}",  # change to a repo name of your choice
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=16 // args.batch_size,  # increase by 2x for every 2x decrease in batch size
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_steps=args.epochs,
        gradient_checkpointing=False,
        evaluation_strategy="steps",
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=750,
        eval_steps=150,
        logging_steps=150,
        report_to="wandb",
        run_name=args.run_name,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        seed=args.seed
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["val"],
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, tokenizer),
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
