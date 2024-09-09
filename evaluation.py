import pandas as pd
import torch
from transformers import  WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, pipeline
from datasets import Dataset, Audio, DatasetDict
import os
from fine_tuning import prepare_dataset, compute_metrics_for_evaluation
from tqdm import tqdm


# for local run we change the cache location
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["HF_HOME"] = "/cs/usr/tomer.cohen13/lab/speach/Project/huggingface/cache"
# os.environ["HF_HUB_CACHE"] = "/cs/usr/tomer.cohen13/lab/speach/Project/huggingface/cache"
# HUG_CACHE = "/cs/usr/tomer.cohen13/lab/speach/Project/huggingface/cache"


# Run evaluation on the test set for whisper small, big, ours after fine-tuning.
# Save all the results (wer only?) in a csv

def get_results(common_voice, model, tokenizer):
    with torch.no_grad():
        predicted_ids = []
        label_ids = []
        for sample_input, sample_label in tqdm(
                zip(common_voice["test"]["input_features"], common_voice["test"]["labels"])):
            predicted_ids.append(model.generate(torch.Tensor(sample_input).unsqueeze(0))[0])
            label_ids.append(torch.Tensor(sample_label))
    results = compute_metrics_for_evaluation(predicted_ids, label_ids, tokenizer)
    return results


def get_metrics_for_baseline(whisper_model):
    # Get data and tokenizer
    data_info_test = pd.read_csv('archive/Dysarthria and Non Dysarthria/Torgo/processed_data_test.csv')
    test_dataset = Dataset.from_pandas(data_info_test).cast_column("Wav_path", Audio(sampling_rate=16000))

    # Map the data to the right format
    common_voice = DatasetDict()
    common_voice["test"] = test_dataset

    # for local run we use cache_dir=HUG_CACHE in WhisperProcessor.from_pretrained()
    processor = WhisperProcessor.from_pretrained(f"openai/{whisper_model}", language="English", task="transcribe")
    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer
    common_voice = common_voice.map(lambda x: prepare_dataset(x, feature_extractor, tokenizer), num_proc=1, remove_columns=common_voice.column_names["test"])

    # Get the desired whisper model
    # for local run we use cache_dir=HUG_CACHE in WhisperForConditionalGeneration.from_pretrained()
    model = WhisperForConditionalGeneration.from_pretrained(f"openai/{whisper_model}")
    model.eval()
    model.generation_config.language = "english"
    model.config.forced_decoder_ids = None
    model.generation_config.task = "transcribe"

    results = get_results(common_voice, model, tokenizer)

    # Save the predictions to a csv
    return results


def get_metrics_for_finetune(whisper_model, checkpoint):

    # Get the desired whisper model
    # for local run we use cache_dir=HUG_CACHE in WhisperTokenizer.from_pretrained()
    tokenizer = WhisperTokenizer.from_pretrained(f"openai/{whisper_model}", language="English", task="transcribe")
    model = pipeline(task="automatic-speech-recognition", model=f"./trained_models/{whisper_model.replace('-v3', '').replace('-', '_')}_fine_tune_clean_wer/checkpoint-{checkpoint}", tokenizer=tokenizer)
    feature_extractor = model.feature_extractor
    tokenizer = model.tokenizer

    # Get data and tokenizer
    data_info_test = pd.read_csv('archive/Dysarthria and Non Dysarthria/Torgo/processed_data_test.csv')
    test_dataset = Dataset.from_pandas(data_info_test).cast_column("Wav_path", Audio(sampling_rate=16000))

    # Map the data to the right format
    common_voice = DatasetDict()
    common_voice["test"] = test_dataset
    common_voice = common_voice.map(lambda x: prepare_dataset(x, feature_extractor, tokenizer), num_proc=1, remove_columns=common_voice.column_names["test"])

    results = get_results(common_voice, model.model, tokenizer)

    # Save the predictions to a csv
    return results


def get_metrics(whisper_model, fine_tune, checkpoint):
    if fine_tune:
        return get_metrics_for_finetune(whisper_model, checkpoint)
    return get_metrics_for_baseline(whisper_model)


if __name__ == '__main__':

    test_metrics_csv = {"Orig_model": [], "Fine_tuned": [], "WER": [], "CER": [], "True": [], "Pred": []}
    test_average_metrics_csv = {"Orig_model": [], "Fine_tuned": [], "WER": [], "CER": []}
    whisper_orig_models = {"whisper-tiny": 3000, "whisper-base": 3000, "whisper-small": 3000, "whisper-medium": 3000, "whisper-large-v3": 3000}

    for whisper_,checkpoint_ in whisper_orig_models.items():

        for is_fine_tuned in [False, True]:
            result_dict = get_metrics(whisper_, is_fine_tuned, checkpoint_)

            test_metrics_csv["Orig_model"] += [whisper_] * len(result_dict["wer"])
            test_metrics_csv["Fine_tuned"] += [is_fine_tuned] * len(result_dict["wer"])
            test_metrics_csv["WER"] += result_dict["wer"]
            test_metrics_csv["CER"] += result_dict["cer"]
            test_metrics_csv["True"] += result_dict["label_str"]
            test_metrics_csv["Pred"] += result_dict["pred_str"]

            test_average_metrics_csv["Orig_model"].append(whisper_)
            test_average_metrics_csv["Fine_tuned"].append(is_fine_tuned)
            test_average_metrics_csv["WER"].append(result_dict["wer_all"])
            test_average_metrics_csv["CER"].append(result_dict["cer_all"])

            print(f"Model: {whisper_}{'_finetune' if is_fine_tuned else ''}, WER: {result_dict['wer_all']}, CER: {result_dict['cer_all']}")

    pd.DataFrame(test_average_metrics_csv).to_csv("test_wer_cer.csv", index=False)
    pd.DataFrame(test_metrics_csv).to_csv("all_test_predictions.csv", index=False)
