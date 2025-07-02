import pandas as pd
from tokenizers import Tokenizer, models, trainers
import gc 
from config import TOKENIZER_PATH, CSV_DATASET_PATH, DESIRED_VOCAB_LEN,\
                    MIN_FREQUENCY, TEXT_COLUMN, CHUNK_SIZE


def csv_text_iterator(csv_file_path, text_column, chunk_size):
    """
    A generator that yields text content from a specified column of a CSV file in chunks.
    This avoids loading the entire CSV into memory.
    """

    try:
        # read the CSV in chunks to save some memory but
        # even if the "counting pairs" thing is still a big issue
        for i, chunk in enumerate(pd.read_csv(csv_file_path, chunksize=chunk_size)):

            for text in chunk[text_column].astype(str).dropna():
                yield text
            
            # explicitly delete chunk and run garbage collection
            del chunk
            gc.collect()

    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
        raise
    except Exception as e:
        print(f"Error reading CSV chunk or processing text: {e}")
        raise


def decode(encoding):
    result = ""
    for e in encoding.tokens:
        result += e
    return result


def train_bpe_tokenizer_on_csv(csv_file, text_column, vocab_size, \
                               min_frequency, tokenizer_name,\
                               chunk_size):

    tokenizer = Tokenizer(models.BPE())

    trainer = trainers.BpeTrainer(vocab_size=vocab_size,\
                                  min_frequency = min_frequency,\
                                  special_tokens = ["<SOS>","<PAD>","<EOS>"])

    tokenizer.train_from_iterator(iterator = csv_text_iterator(csv_file, text_column, chunk_size),
                                  trainer = trainer)

    tokenizer.save(TOKENIZER_PATH, pretty=True)

    print(f"tokenizer saved to: {TOKENIZER_PATH}")

    return tokenizer


try:

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

except Exception as e:

    if "No such file or directory" in str(e):

        print(f"can't load {TOKENIZER_PATH}, training a new tokenizer")

        tokenizer = train_bpe_tokenizer_on_csv(
                        csv_file = CSV_DATASET_PATH,
                        text_column = TEXT_COLUMN,
                        vocab_size = DESIRED_VOCAB_LEN ,
                        min_frequency = MIN_FREQUENCY,
                        tokenizer_name = TOKENIZER_PATH,
                        chunk_size = CHUNK_SIZE)

if __name__ == "__main__":

    test_sentence  = "<SOS>questa è una frase di prova<EOS><PAD><PAD>"
    print(f"Original sentence: {test_sentence}")

    encoding = tokenizer.encode(test_sentence)

    print("Encoded tokens:", encoding.tokens)
    print("Encoded IDs:", encoding.ids)
    print("Decoded text:", decode(encoding))
