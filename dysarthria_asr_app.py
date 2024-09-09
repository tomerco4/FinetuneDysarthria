from transformers import pipeline
import gradio as gr
from transformers import  WhisperTokenizer
from datasets import Dataset, Audio, DatasetDict
import pandas as pd
from fine_tuning import prepare_dataset, clean_string
import torch


# HUG_CACHE = "/cs/usr/tomer.cohen13/lab/speach/Project/huggingface/cache"
# for local run we use cache_dir=HUG_CACHE in WhisperTokenizer.from_pretrained()

tokenizer = WhisperTokenizer.from_pretrained(f"openai/whisper-small", language="English", task="transcribe")
pipe = pipeline(task="automatic-speech-recognition", model="trained_models/whisper_small_fine_tune_clean_wer/checkpoint-3000", tokenizer=tokenizer)  # change to "your-username/the-name-you-picked"
feature_extractor = pipe.feature_extractor
tokenizer = pipe.tokenizer


def transcribe(audio):
    df = pd.DataFrame({"Wav_path": [audio], "Prompts": [""]})
    dataset = Dataset.from_pandas(df).cast_column("Wav_path", Audio(sampling_rate=16000))
    common_voice = DatasetDict()
    common_voice["sample"] = dataset
    common_voice = common_voice.map(lambda x: prepare_dataset(x, feature_extractor, tokenizer), num_proc=1, remove_columns=common_voice.column_names["sample"])
    pred_ids = pipe.model.generate(torch.Tensor(common_voice["sample"]["input_features"][0]).unsqueeze(0))
    pred_str = [clean_string(s) for s in tokenizer.batch_decode(pred_ids, skip_special_tokens=True)][0]
    return pred_str


iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Microphone(type="filepath"),
    outputs="text",
    title="Whisper Small Dysarthria",
    description="Realtime demo for Dysarthria speech recognition using a fine-tuned Whisper small model.",
)

iface.launch()