# FinetuneDysarthria

Here’s a sample `README.md` file for your GitHub repository based on the files you uploaded:

```markdown
# Fine-Tuning Whisper Model for ASR on Dysarthric Speech

This repository contains code to fine-tune the Whisper model for Automatic Speech Recognition (ASR) on dysarthric speech data, such as the TORGO dataset. The project includes scripts for data processing, model training, and evaluation, as well as a web application for deploying the ASR model.

## Project Structure

```
.
├── fine_tuning.py        # Script for fine-tuning Whisper models on dysarthric speech data
├── dysarthria_asr_app.py # A web application for ASR demo on dysarthric speech
├── process_data.py       # Data pre-processing and preparation for the ASR model
├── plot.py               # Plotting and visualization of model performance
└── README.md             # Project documentation (this file)
```

## Requirements

Before running any of the scripts, make sure to install the necessary dependencies. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

If a `requirements.txt` file does not exist, make sure to manually install these dependencies:

- `torch`
- `transformers`
- `datasets`
- `librosa`
- `matplotlib`
- `seaborn`
- `flask` (for web app deployment)
- `pydub`
- `pandas`

## Files Description

### `fine_tuning.py`

This script is responsible for fine-tuning the pre-trained Whisper ASR models on dysarthric speech datasets. It includes the training loop, evaluation metrics such as Word Error Rate (WER) and Character Error Rate (CER), and logging.

#### Usage:

```bash
python fine_tuning.py --dataset_path <path-to-dataset> --model_name whisper-small
```

### `dysarthria_asr_app.py`

This script deploys a simple Flask web application that allows users to upload audio files and receive transcription using the fine-tuned Whisper model.

#### Usage:

```bash
python dysarthria_asr_app.py
```

Then, navigate to `http://localhost:5000` in your browser to use the app.

### `process_data.py`

This script handles the pre-processing of the dysarthric speech dataset, such as converting audio to a compatible format, extracting features, and preparing data for model training.

#### Usage:

```bash
python process_data.py --dataset_path <path-to-raw-data>
```

### `plot.py`

This script generates various visualizations of the model’s performance, including bar plots and box plots of WER and CER across different Whisper model variants.

#### Usage:

```bash
python plot.py
```

This will output figures to the `figures/` directory.

## Dataset

The main dataset used in this project is the [TORGO dataset](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html), which contains recordings of dysarthric and control speakers.

Ensure the dataset is preprocessed and formatted correctly using `process_data.py` before training the models.

## Results

- The fine-tuned models achieved significant improvements in WER and CER compared to the pre-trained versions.
- Detailed visualizations can be found in the `figures/` directory after running the `plot.py` script.

## Running the Web App

To run the Flask web application for testing your fine-tuned ASR model on dysarthric speech, simply use:

```bash
python dysarthria_asr_app.py
```

Once running, visit `http://localhost:5000` to interact with the web interface and upload audio files for transcription.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

### Key Components of the README:

1. **Project Structure**: Lists the key files in the project.
2. **Requirements**: Includes installation of necessary packages.
3. **Files Description**: Provides brief descriptions and usage instructions for each file.
4. **Dataset**: Mentions the dataset used (TORGO dataset) for the dysarthric speech.
5. **Running the Web App**: Instructions for running the Flask application.
6. **License**: Placeholder for licensing info.

Let me know if you need any more details or adjustments!
