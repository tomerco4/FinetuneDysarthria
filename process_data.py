import pandas as pd
import os

torgo_dir = "archive/Dysarthria and Non Dysarthria/Torgo"
os.chdir(torgo_dir)

data_csv = {"Wav_path": [], "Txt_path": [], "Prompts": []}

for gender in ["F", "M"]:
    os.chdir(gender)
    for subject in [d for d in os.listdir(".") if os.path.isdir(d)]:
        os.chdir(subject)
        for session in [d for d in os.listdir(".") if "Session" in d]:
            os.chdir(session)

            wav_dir = "wav_headMic" if os.path.exists("wav_headMic") else "wav_arrayMic"
            txt_dir = "prompts"

            for prompt_path in os.listdir(txt_dir):
                with open(os.path.join(txt_dir, prompt_path), "r") as f:
                    prompt = " ".join(f.readlines()).replace("\n", "").strip()

                if "xxx" in prompt or prompt.startswith("[") or ".jpg" in prompt:
                    continue
                if prompt.endswith(".") or prompt.endswith(";"):
                    prompt = prompt[:-1]
                if "[" in prompt:
                    prompt = prompt[:prompt.find("[")] + prompt[prompt.find("]") + 1:]
                    prompt = prompt.strip()
                print(f"{prompt}")

                wav_path = prompt_path.replace(".txt", ".wav")

                if not os.path.exists(os.path.join(wav_dir, wav_path)):
                    continue

                data_csv["Wav_path"].append(os.path.join(torgo_dir, gender, subject, session, wav_dir, wav_path))
                data_csv["Txt_path"].append(os.path.join(torgo_dir, gender, subject, session, txt_dir, prompt_path))
                data_csv["Prompts"].append(prompt)

            os.chdir("..")
        os.chdir("..")

    os.chdir("..")


pd.DataFrame(data_csv).to_csv("processed_data.csv", index=False)



