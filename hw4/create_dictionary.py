"""
Converts the OpenRussian.org raw CSVs into a single English-Russian dictionary for translation

Output format: en_word <tab> ru_word
"""
import pandas as pd
import numpy as np
import re


words_path = "openrussian-csv/words.csv"
fixed_words_path = "openrussian-csv/my_words.csv"
translation_path = "openrussian-csv/translations.csv"
output_dict = "my_en_ru_dict.tsv"

# Read words and translations into Pandas
# Fix words file first - dictionary entries roll over to new line and messes up the parsing
with open(words_path) as f:
    lines = f.readlines()
with open(fixed_words_path, "w+") as f:
    f.write(lines[0])
    for line in lines:
        if re.match("^\d", line[0]):
            f.write(line)

words_df = pd.read_csv(fixed_words_path, delimiter="\t", header=0, index_col=0, dtype={"id": np.int, "position": pd.Int64Dtype()})
translation_df = pd.read_csv(translation_path,
                             delimiter="\t",
                             header=0,
                             index_col=0)

# Drop columns and data we don't need to save space
translation_df = translation_df[translation_df.lang == "en"][["word_id", "position", "tl"]]
translation_df.word_id = translation_df.word_id.astype(int)
translation_df.tl = translation_df.tl.map(lambda x: [i.strip() for i in x.split(",")])
words_df = words_df[["position", "bare", "accented"]]


def get_ru_word(id):
    return words_df.loc[id].accented

# Join translations and words based on ID
translation_df["ru_word"] = translation_df.word_id.map(lambda x: get_ru_word(x))

# Print out dictionary in TSV form
with open(output_dict, "w+") as f:
    for row in translation_df.itertuples():
        for en_translation in row.tl:
            #f.write(f"{en_translation}\t{row.ru_word}\n")
            f.write(f"{row.ru_word}\t{en_translation}\n")

