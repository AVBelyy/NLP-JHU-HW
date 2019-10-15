"""
Projects information from an English sentence to a sentence from another language

Runs as follows:
python crosslingual.py mygrammar.gr eng-deu.dict eng.sen deu.sen
"""

import argparse
from parse_ext import CKYParser
import re


class ParseNotFoundError(Exception):
    pass


class GrammarProjector:
    def __init__(self, grammar_path, dictionary_path):
        # Initialize internal parser
        self.parser = CKYParser("BEST-PARSE", grammar_path)

        # Read in bilingual dictionary
        self.bilingual_dict = self.read_in_dictionary(dictionary_path)


    @staticmethod
    def read_in_dictionary(dictionary_path):
        bilingual_dict = {}
        with open(dictionary_path) as f:
            for line in f.readlines():
                english_token, foreign_token = [token.strip() for token in line.split("\t")]
                if english_token in bilingual_dict:
                    bilingual_dict[english_token].append(foreign_token)
                else:
                    bilingual_dict[english_token] = [foreign_token]
        return bilingual_dict

    def project_grammar(self, en_sentence, foreign_sentence):
        weight, back_pointers = self.parser.parse_sentence(en_sentence)

        if weight is None:
            raise ParseNotFoundError
        inv_grammar = self.get_preterminal_mappings(en_sentence, back_pointers)

        projection = {token: "-" for token in foreign_sentence.split()}
        for token in en_sentence.split():
            if token in self.bilingual_dict:
                for translation in self.bilingual_dict[token]:
                    if translation in projection:
                        projection[translation] = inv_grammar[token]
        projection_result = "\t".join([projection[token] for token in foreign_sentence.split()])
        return projection_result

    def get_preterminal_mappings(self, sentence, backpointers):
        mappings = {}
        # Get tree representation and parse for pre-terminals
        tree = self.parser.generate_tree(sentence, backpointers)
        endings = [i.strip() for i in str(tree).split(")")]

        for ending in endings:
            if ending == "": continue
            ending = re.sub("\(", "", ending).split(" ")
            mappings[ending[-1]] = ending[-2]
        return mappings


def parse_arguments():
    parser = argparse.ArgumentParser(description="Project information from an English sentence "
                                                 "to a foreign language sentence using word alignments")
    parser.add_argument("grammar", help="Path to grammar file")
    parser.add_argument("dictionary", help="English-Foreign dictionary path. "
                                           "Must match the language of the foreign language sentences")
    parser.add_argument("en_sentences", help="Path to file containing the English sentences to parse")
    parser.add_argument("f_sentences", help="Path to file containing the foreign-language sentences to parse")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    projector = GrammarProjector(args.grammar, args.dictionary)

    with open(args.en_sentences) as f:
        en_sentences = [i.strip() for i in f.readlines()]
    with open(args.f_sentences) as f:
        foreign_sentences = [i.strip() for i in f.readlines()]

    for en_sentence, foreign_sentence in zip(en_sentences, foreign_sentences):
        try:
            p = projector.project_grammar(en_sentence.lower(), foreign_sentence.lower())
            print(foreign_sentence.replace(' ', '\t'))
            print(p)
        except ParseNotFoundError as err:
            print(f"Error: No parse found for input sentence '{en_sentence}'")

