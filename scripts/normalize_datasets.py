import pandas as pd


def join_para_text(x):
    # print(x)
    options = eval(x["choices"])

    return f"Question: {x['paraphrased_question']}Option A: {options[0]}\nOption B: {options[1]}\nOption C: {options[2]}\nOption D: {options[3]}"

def join_input_text(x):
    # print(x)
    options = eval(x["choices"])

    return f"Question: {x['question']}\nOption A: {options[0]}\nOption B: {options[1]}\nOption C: {options[2]}\nOption D: {options[3]}"


# Define file names
# source_file = "train_gemma_dpo.csv"
# destination_file = "train_gemma_dpo.csv"

source_file = "test_gemma_dpo.csv"
destination_file = "test_gemma_dpo.csv"


# Specify the column name or index to copy
column_to_copy = (
    "paraphrased_question"  # Replace with actual column name or use an index (e.g., 1)
)

# Read the source CSV and extract the column
source_df = pd.read_csv(source_file)

# Read the destination CSV
destination_df = pd.read_csv(destination_file)

# Ensure the column exists in the source file
if column_to_copy not in source_df.columns:
    raise KeyError(f"Column '{column_to_copy}' not found in {source_file}")

# Add the column to the destination file
destination_df[column_to_copy] = source_df[column_to_copy]

if "input_text" not in destination_df:
    destination_df["input_text"] = destination_df.apply(join_input_text, axis=1)

destination_df = destination_df[
    [
        "question",
        "paraphrased_question",
        "subject",
        "choices",
        "answer",
        "input_text",
    ]
]

destination_df["paraphrased_text"] = destination_df.apply(join_para_text, axis=1)

# Save the updated destination file
destination_df.to_csv("test_gemma_dpo_new.csv", index=False)

print("Column copied successfully!")
