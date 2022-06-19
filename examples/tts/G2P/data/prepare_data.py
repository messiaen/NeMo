import json
import re
from typing import List, Optional

from nemo_text_processing.text_normalization.normalize import Normalizer
from tqdm import tqdm

from nemo.collections.tts.torch.en_utils import english_word_tokenize
from nemo.collections.tts.torch.g2ps import IPAG2P


class IPAG2PProcessor(IPAG2P):
    def __init__(
        self,
        phoneme_dict,
        word_tokenize_func=english_word_tokenize,
        ignore_ambiguous_words=True,
        heteronyms=None,
        grapheme_unk_token="҂",
        ipa_unk_token="҂",
        phoneme_probability: Optional[float] = None,
        use_stresses: Optional[bool] = True,
        set_graphemes_upper: Optional[bool] = True,
    ):

        self.grapheme_unk_token = grapheme_unk_token
        self.ipa_unk_token = ipa_unk_token

        super().__init__(
            phoneme_dict=phoneme_dict,
            word_tokenize_func=word_tokenize_func,
            apply_to_oov_word=lambda x: self.ipa_unk_token,
            ignore_ambiguous_words=ignore_ambiguous_words,
            heteronyms=heteronyms,
            phoneme_probability=phoneme_probability,
            use_stresses=use_stresses,
            set_graphemes_upper=set_graphemes_upper,
        )

    def post_process(self, symbols: List[str]):
        return "".join(symbols)

    def parse_one_word(self, word: str):
        """
		Returns parsed `word` and `status` (bool: False if word wasn't handled, True otherwise).
		"""
        if self.set_graphemes_upper:
            word = word.upper()

        if self.phoneme_probability is not None and self._rng.random() > self.phoneme_probability:
            return word, True

        # Punctuation (assumes other chars have been stripped)
        if re.search("[a-zA-Z]", word) is None:
            return list(word), True

        # Heteronym
        if self.heteronyms and word in self.heteronyms:
            return self.apply_to_oov_word(word), False

        # `'s` suffix (with apostrophe) - not in phoneme dict
        if (
            len(word) > 2
            and word.endswith("'s")
            and (word not in self.phoneme_dict)
            and (word[:-2] in self.phoneme_dict)
            and (not self.ignore_ambiguous_words or self.is_unique_in_phoneme_dict(word[:-2]))
        ):
            if word[-3] == 'T':
                # Case like "airport's"
                return self.phoneme_dict[word[:-2]][0] + ["t", "s"], True
            elif word[-3] == 'S':
                # Case like "jones's"
                return self.phoneme_dict[word[:-2]][0] + ["ɪ", "z"], True
            else:
                return self.phoneme_dict[word[:-2]][0] + ["z"], True

        # `s` suffix (without apostrophe) - not in phoneme dict
        if (
            len(word) > 1
            and word.endswith("s")
            and (word not in self.phoneme_dict)
            and (word[:-1] in self.phoneme_dict)
            and (not self.ignore_ambiguous_words or self.is_unique_in_phoneme_dict(word[:-1]))
        ):
            if word[-2] == 'T':
                # Case like "airports"
                return self.phoneme_dict[word[:-1]][0] + ["t", "s"], True
            elif word.endswith("IES"):
                # Case like "flies", "brandies", don't handle -> multiple possible IPA endings
                return self.apply_to_oov_word(word), False
            else:
                return self.phoneme_dict[word[:-1]][0] + ["z"], True

        # Phoneme dict lookup for unique words (or default pron if ignore_ambiguous_words=False)
        if word in self.phoneme_dict and (not self.ignore_ambiguous_words or self.is_unique_in_phoneme_dict(word)):
            return self.phoneme_dict[word][0], True

        # use ipa_unk_token for OOV words
        return self.apply_to_oov_word(word), False

    def __call__(self, text):
        words = self.word_tokenize_func(text)

        prons = []
        graphemes = []
        for word, without_changes in words:
            pron, phon_is_handled = self.parse_one_word(word)

            if phon_is_handled:
                graphemes.append(word)
            else:
                word_by_hyphen = word.split("-")
                if len(word_by_hyphen) > 1:
                    pron = []
                    for i, sub_word in enumerate(word_by_hyphen):
                        p, phon_is_handled = self.parse_one_word(sub_word)
                        pron.extend(p)
                        if phon_is_handled:
                            graphemes.append(sub_word)
                        else:
                            assert p == self.ipa_unk_token
                            graphemes.append(self.grapheme_unk_token)

                        if i < (len(word_by_hyphen) - 1):
                            pron.extend(["-"])
                            graphemes.append("-")

                else:
                    assert pron == self.ipa_unk_token
                    graphemes.append(self.grapheme_unk_token)

            prons.extend(pron)

        prons = self.post_process(prons)
        graphemes = self.post_process(graphemes)
        return prons, graphemes


if __name__ == "__main__":
    ipa_tok = IPAG2PProcessor(
        phoneme_dict="/home/ebakhturina/NeMo/scripts/tts_dataset_files/ipa_cmudict-0.7b_nv22.06.txt",
        heteronyms="/home/ebakhturina/NeMo/scripts/tts_dataset_files/heteronyms-052722",
        grapheme_unk_token="҂",
        ipa_unk_token="҂",
        ignore_ambiguous_words=False,
        set_graphemes_upper=False,
        use_stresses=True,
    )

    text = "how to use the contents?"
    gt_ipa = "ˈhaʊ ˈtu ҂ ðə ˈkɑntɛnts?"
    gt_graphemes = "how to ҂ the contents?"
    pred_phons, pred_graphemes = ipa_tok(text)
    assert pred_phons == gt_ipa and pred_graphemes == gt_graphemes

    text = "hello-world-waveglow!"
    gt_ipa = "həˈɫoʊ-ˈwɝɫd-ˈweɪvˌɡɫoʊ!"
    gt_graphemes = "hello-world-waveglow!"
    pred_phons, pred_graphemes = ipa_tok(text)
    assert pred_phons == gt_ipa and pred_graphemes == gt_graphemes

    for subset in ["train", "eval"]:
        from glob import glob
        from tqdm import tqdm
        import csv

        # from nemo.collections.tts.torch.g2p_classification_data import read_wikihomograph_file

        # normalized data and saves in "WikipediaHomographData-master/data/{subset}_normalized"
        # is output file is present, skips normalization
        # normalize_wikihomograph_data(subset)

        def read_wikihomograph_normalized_file(file: str) -> (List[str], List[List[int]], List[str], List[str]):
            """
            Reads .tsv file from WikiHomograph dataset,
            e.g. https://github.com/google-research-datasets/WikipediaHomographData/blob/master/data/eval/live.tsv

            Args:
                file: path to .tsv file
            Returns:
                sentences: Text.
                start_end_indices: Start and end indices of the homograph in the sentence.
                homographs: Target homographs for each sentence (TODO: check that multiple homograph for sent are supported).
                word_ids: Word_ids corresponding to each homograph, i.e. label.
            """
            normalized_sentences = []
            word_ids = []
            with open(file, "r", encoding="utf-8") as f:
                tsv_file = csv.reader(f, delimiter="\t")
                for i, line in enumerate(tsv_file):
                    if i == 0:
                        continue
                    _, wordid, _, sentence_normalized = line
                    normalized_sentences.append(sentence_normalized)
                    word_ids.append(wordid)
            return normalized_sentences, word_ids

        normalized_data = f"/home/ebakhturina/g2p_scripts/WikipediaHomographData-master/data/{subset}_normalized/"
        files = glob(f"{normalized_data}/*.tsv")

        manifest = f"/mnt/sdb_4/g2p/data_ipa/with_unicode_token/{subset}_wikihomograph.json"
        with open(manifest, "w", encoding="utf-8") as f_out:
            for file in tqdm(files):
                sentences, word_ids = read_wikihomograph_normalized_file(file)
                for idx, sent in enumerate(sentences):
                    ipa_, graphemes_ = ipa_tok(sent)
                    entry = {
                        "text": ipa_,
                        "text_graphemes": graphemes_,
                        "wordid": word_ids[idx],
                        "duration": 0.001,
                        "audio_filepath": "n/a",
                    }

                    f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Data for {subset.upper()} saved at {manifest}")

    # with open("test.json", "r") as f:
    #     for line in f:
    #         line = json.loads(line)
    #         text = line["text_graphemes"]
    #         ipa = line["text"]
    #         prons, graphemes = ipa_tok(text)
    #         print("=" * 40)
    #         print(f"GT   -> {text}")
    #         print(f"PRED -> {graphemes}\n")
    #         print(f"IPA GT   -> {ipa}")
    #         print(f"IPA PRED -> {prons}")
    #
    # print("".join(ipa_tok("Hello, world!")))
