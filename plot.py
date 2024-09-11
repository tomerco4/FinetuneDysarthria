import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV files
file_path = 'all_test_predictions.csv'
all_df = pd.read_csv(file_path)
file_path = 'test_wer_cer.csv'
df = pd.read_csv(file_path)
all_df_filter = all_df[all_df["WER"] <= 700].reset_index(drop=True)
all_df_filter = all_df_filter[all_df_filter["CER"] <= 700].reset_index(drop=True)


# remove whisper from name
all_df["Orig_model"] = [i.replace("whisper-", "") for i in all_df["Orig_model"]]
all_df_filter["Orig_model"] = [i.replace("whisper-", "") for i in all_df_filter["Orig_model"]]
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
        sns.violinplot(x="Orig_model", y=metric, hue="Fine_tuned", color="Fine_tuned", data=all_df_filter, palette=palette, split=True, inner="quart", gap=.1)
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


def plot_scatter_fine_tune_vs_regular():
    # Filter the data for the whisper-small model
    small_model_df = all_df[all_df["Orig_model"] == "small"]

    # Separate the fine-tuned and regular (non-fine-tuned) data
    fine_tuned_df = small_model_df[small_model_df["Fine_tuned"] == True].reset_index(drop=True)
    regular_df = small_model_df[small_model_df["Fine_tuned"] == False].reset_index(drop=True)

    for metric in metrics:
        plt.figure(figsize=(10, 10))
        # Scatter plot comparing regular vs fine-tuned
        plt.scatter(fine_tuned_df[metric], regular_df[metric], color="yellowgreen", s=100, alpha=0.5)

        same = sum([x == y for x,y in zip(fine_tuned_df[metric], regular_df[metric])])
        better = sum([x < y for x,y in zip(fine_tuned_df[metric], regular_df[metric])])
        worse = sum([x > y for x,y in zip(fine_tuned_df[metric], regular_df[metric])])

        print(f"{metric}: Same: {same}, Better: {better}, Worse: {worse}")

        # Plot the x=y line for reference

        min_val, max_val = -10, 350
        plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='black', label="x=y")

        plt.xlim(min_val, max_val)  # You can specify a range like (0, 100) if you prefer fixed limits
        plt.ylim(min_val, max_val)

        plt.ylabel(f"Whisper-small {metric} (%)")
        plt.xlabel(f"Whisper-small Fine-tuned {metric} (%)")
        # plt.title(f"Scatter Plot of Regular vs Fine-tuned {metric} (whisper-small)")
        plt.grid(True)
        #plt.legend()
        plt.savefig(f"figures/scatter_fine_tuned_vs_regular_{metric}.png".lower(), dpi=500)


if __name__ == '__main__':
    plot_bars()
    plot_boxplot()
    plot_zero_metrics_percentage()
    plot_scatter_fine_tune_vs_regular()