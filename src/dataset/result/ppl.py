import argparse
from evaluate import load
import pandas as pd


def main(test_file, test_column):
    perplexity = load("perplexity", module_type="metric")
    df = pd.read_csv(test_file)
    print(df['completions'].tolist())
    cleaned_sentences = df['completions'].tolist()
    #cleaned_sentences = cleaned_sentences[:983] + cleaned_sentences[992:]

    results = perplexity.compute(model_id='gpt2-large',
                                 add_start_token=False,
                                 predictions=cleaned_sentences)

    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some inputs.")
    parser.add_argument("--test_file", type=str, required=True, help="file name")
    parser.add_argument("--test_column", type=str, default="model_real_output", help="test column")

    args = parser.parse_args()
    main(args.test_file, args.test_column)
