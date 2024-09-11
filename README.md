## Dysarthria Speech Recognition (ASR) Project

This project aims to build an Automatic Speech Recognition (ASR) system tailored for dysarthric speech using a fine-tuned Whisper model. The ASR system is implemented using Python and HuggingFace's `transformers` library, and provides a real-time demo using Gradio.
We based our code on the "Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers" tutorial (https://huggingface.co/blog/fine-tune-whisper)


<p align="center"><img src="https://github.com/tomerco4/FinetuneDysarthria/blob/main/figures/Logo.webp" width="500" /></p>


### Features

- Fine-tuning of OpenAI's Whisper models for improved recognition of dysarthric speech.
- Real-time speech recognition demo using Gradio.
- Evaluation of model performance using Word Error Rate (WER) and Character Error Rate (CER).
- Data preprocessing and analysis scripts for the TORGO dataset.

### Installation

To set up the project, ensure you have Python installed, then clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/dysarthria-asr.git
cd dysarthria-asr
pip install -r requirements.txt
```

### Usage

#### 1. Running the Real-time Demo

To run the real-time ASR demo, execute the following command:

```bash
python dysarthria_asr_app.py
```

This will launch a Gradio interface in your browser, where you can use your microphone to test the model's transcription capabilities.

#### 2. Fine-tuning the Whisper Model

To fine-tune a Whisper model on your dataset, use the following script:

```bash
python fine_tuning.py --model whisper-small --epochs 5000 --batch_size 16 --learning_rate 1e-5 --run_name "your-run-name"
```

Additional parameters such as `learning_rate`, `batch_size`, and `epochs` can be adjusted as needed.

#### 3. Evaluation

You can evaluate the performance of the trained model using:

```bash
python evaluation.py
```

The evaluation results will be saved to CSV files (`test_wer_cer.csv` and `all_test_predictions.csv`) for further analysis.

#### 4. Plotting Results

To visualize the performance metrics, run:

```bash
python plot.py
```

This script will generate plots for WER and CER metrics, which will be saved in the `figures` directory.

### Data Preparation

The project includes scripts to process and split the TORGO dataset into training, validation, and test sets:

- `process_data.py`: Prepares the data by extracting audio and text prompts.
- `train_val_test_split.py`: Splits the data into training, validation, and test sets.

### Requirements

The project requires the following Python packages:

- `transformers==4.44.0`
- `datasets==2.21.0`
- `gradio==4.41.0`
- `torch==1.13.1`
- `torchaudio==0.13.1`
- Other dependencies listed in `requirements.txt`

