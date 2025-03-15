

import os
import sys


def read_afinn(path):
    """
    Reads the AFINN lexicon from the given path and returns a dictionary of words and their sentiment scores.

    :param path: Path to the AFINN lexicon file
    :return: Dictionary with words as keys and sentiment scores as integer values
    """
    afinn_dict = {}
    
    # Open the file safely
    with open(path, 'r', encoding='latin1') as flexicon:
        for line in flexicon:
            line = line.strip()
            if line:
                # Try splitting by tab first, then fallback to whitespace
                if '\t' in line:
                    items = line.split('\t')
                else:
                    items = line.split()  # Fallback to whitespace

                # Ensure there are exactly 2 items (word and score)
                if len(items) == 2:
                    word = items[0]
                    try:
                        score = int(items[1])  # Convert score to integer
                        afinn_dict[word] = score
                    except ValueError:
                        print(f"Warning: Unable to parse score for line: {line}")
                else:
                    print(f"Warning: Skipping invalid line: {line}")

    return afinn_dict


if __name__ == '__main__':
    # Update with the actual path to your AFINN file
    afinn_path = "SentimentLexicons/AFINN-111.txt"
    
    # Load the AFINN lexicon
    try:
        afinn = read_afinn(afinn_path)
    except FileNotFoundError:
        print(f"Error: File not found at '{afinn_path}'. Please check the path and try again.")
        sys.exit(1)

    # Words to check
    words = ['attracting', 'attraction', 'avoids', 'axe', 'bad']

    # Get and print scores
    for word in words:
        score = afinn.get(word, "Not found")  # Default to "Not found" if the word isn't in the lexicon
        print(f"The AFINN score of the word '{word}' is: {score}")
