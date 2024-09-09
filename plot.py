import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV files
file_path = 'all_test_predictions.csv'
all_df = pd.read_csv(file_path)
file_path = 'test_wer_cer.csv'
df = pd.read_csv(file_path)
all_df = all_df[all_df["WER"] <= 700].reset_index(drop=True)
all_df = all_df[all_df["CER"] <= 700].reset_index(drop=True)


# remove whisper from name
all_df["Orig_model"] = [i.replace("whisper-", "") for i in all_df["Orig_model"]]
df["Orig_model"] = [i.replace("whisper-", "") for i in df["Orig_model"]]

palette = ["tomato", "cornflowerblue"]
metrics = ["WER", "CER"]
sns.set(font_scale=2)


def plot_bars():
    for metric in metrics:
        # Plot of entire test average WER (bar plot)
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Orig_model", y=metric, hue="Fine_tuned", color="Fine_tuned", data=df, palette=palette, width=0.6)
        #plt.title(F'Average {metric} by Model')
        plt.ylabel(f"{metric} (%)")
        plt.xlabel('')
        plt.legend(title='Fine Tuned')
        plt.savefig(f"figures/{metric}_bar_plot.png".lower(), dpi=500)


def plot_boxplot():
    for metric in metrics:
        # Plot of entire test WER (boxplot)
        plt.figure(figsize=(10, 6))
        sns.violinplot(x="Orig_model", y=metric, hue="Fine_tuned", color="Fine_tuned", data=all_df, palette=palette, split=True, inner="quart", gap=.1)
        #plt.title(f'{metric} Distribution by Model')
        plt.ylabel(f"{metric} (%)")
        plt.xlabel('')
        plt.legend(title='Fine Tuned')
        # plt.ylim(-100, 600)
        plt.savefig(f"figures/{metric}_box_plot.png".lower(), dpi=500)


def plot_zero_metrics_percentage():
    x_order = ['tiny', 'base', 'small', 'medium', 'large-v3']

    for metric in metrics:
        # Calculate percentage of test cases where WER or CER is 0
        zero_metric_df = all_df.groupby(["Orig_model", "Fine_tuned"]).apply(
            lambda x: (x[metric] == 0).sum() / len(x) * 100).reset_index(name=f"Zero_{metric}_Percentage")

        # Plot the percentage of zero WER/CER test cases
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Orig_model", y=f"Zero_{metric}_Percentage", hue="Fine_tuned", color="Fine_tuned", data=zero_metric_df, palette=palette, width=0.6, order=x_order)
        plt.ylabel(f"Zero {metric} Cases (%)")
        plt.xlabel('')
        plt.legend(title='Fine Tuned')
        plt.savefig(f"figures/zero_{metric}_percentage_plot.png".lower(), dpi=500)


if __name__ == '__main__':
    plot_bars()
    plot_boxplot()
    plot_zero_metrics_percentage()