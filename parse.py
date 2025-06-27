import pandas as pd
import re


def split_models(text):
    model_blocks = text.split("T.1")

    return model_blocks


def parse_model_data(text):
    model_data = {}

    subaspect_pattern = r"The subaspect of (.*?)'s score is ([0-9.]+)"
    subaspects = re.findall(subaspect_pattern, text)
    for subaspect, score in subaspects:
        model_data[f"Subaspect_{subaspect}"] = float(score)

    total_score_pattern = r"The total score of (.*?) is ([0-9.]+)"
    total_score = re.search(total_score_pattern, text)
    if total_score:
        model_data["Model"] = total_score.group(1)
        model_data["Total_Score"] = float(total_score.group(2))

    rank_pattern = r"The rank of .*? is ([0-9]+)"
    rank = re.search(rank_pattern, text)
    if rank:
        model_data["Rank"] = int(rank.group(1))

    return model_data


def parse_text(text):

    model_blocks = split_models(text)

    results = []
    for block in model_blocks:
        if block.strip():
            model_data = parse_model_data(block)
            if model_data:
                results.append(model_data)

    df = pd.DataFrame(results)

    cols = ["Model", "Total_Score", "Rank"] + [
        col for col in df.columns if col.startswith("Subaspect_")
    ]
    df = df[cols]

    return df


with open("score.log", "r") as fp:
    text = fp.read()

rename_mapping = {
    
    'Model': 'model',
    'Subaspect_Inherent Deficiency in Truthfulness': 'T.I',
    'Subaspect_Misguided Mistakes in Truthfulness': 'T.M',
    'Subaspect_Toxicity in Safety': 'S.T', 
    'Subaspect_Jailbreaking in Safety': 'S.J',
    'Subaspect_OOD in Robustness': 'R.O',
    'Subaspect_Adversarial Attack in Robustness': 'R.A',
    'Subaspect_Stereotype in Fairness': 'F.S',
    'Subaspect_Bias & Preference in Fairness': 'F.B',
    'Subaspect_Privacy Awareness in Privacy': 'P.A',
    'Subaspect_Privacy Leakage in Privacy': 'P.L',
    'Total_Score': 'Avg.', 
    'Rank': 'Rank',
}

cols = list(rename_mapping.keys())

result_df = parse_text(text)
result_df = result_df[cols]
result_df = result_df.rename(columns=rename_mapping)

print(result_df)


result_df.to_csv("model_scores.csv", index=False)
