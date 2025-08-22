import gzip
import html
import os
import re
import typing
from functools import lru_cache


@lru_cache()
def default_bpe() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data/bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode() -> typing.Dict[int, str]:
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    byte_ints = list(range(ord("!"),
                           ord("~") + 1)) + list(range(ord("¡"),
                                                       ord("¬") + 1)) + list(
                                                           range(
                                                               ord("®"),
                                                               ord("ÿ") + 1))
    char_ints = byte_ints[:]
    n = 0
    for b in range(2**8):
        if b not in byte_ints:
            byte_ints.append(b)
            char_ints.append(2**8 + n)
            n += 1
    chars = [chr(n) for n in char_ints]
    return dict(zip(byte_ints, chars))


def get_pairs(
        word: typing.Tuple[str, ...]) -> typing.Set[typing.Tuple[str, str]]:
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text: str) -> str:
    import ftfy

    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class SimpleTokenizer(object):

    def __init__(self, bpe_path: str = default_bpe()) -> None:
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        with gzip.open(bpe_path) as f:
            lines = f.read().decode("utf-8").split("\n")
            lines = lines[1:49152 - 256 - 2 + 1]
        merges = [tuple(line.split()) for line in lines]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])
        self.encoder: typing.Dict[str,
                                  int] = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>"
        }
        pattern = r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"""
        pattern = pattern.replace(r"\p{N}", read_text("llm/clip/data/pN.txt"))
        pattern = pattern.replace(r"\p{L}", read_text("llm/clip/data/pL.txt"))
        self.pat = re.compile(pattern, re.IGNORECASE)

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>", )
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(
                pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: typing.List[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[
                        i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        joined_word = " ".join(word)
        self.cache[token] = joined_word
        return joined_word

    def encode(self,
               text: str,
               basic_cleaning: bool = False) -> typing.List[int]:
        bpe_tokens: typing.List[int] = []
        if basic_cleaning:
            text = basic_clean(text)
        text = whitespace_clean(text).lower()
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b]
                            for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token]
                              for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def decode(self, tokens: typing.Iterable[int]) -> str:
        text = "".join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text
                          ]).decode("utf-8",
                                    errors="replace").replace("</w>", " ")
        return text
