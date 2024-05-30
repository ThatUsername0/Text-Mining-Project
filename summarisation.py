from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Initialize the tokenizer and model for long-t5 with specified model_max_length
tokenizer = AutoTokenizer.from_pretrained("google/long-t5-tglobal-base", model_max_length=7500, legacy=False)
model = AutoModelForSeq2SeqLM.from_pretrained("google/long-t5-tglobal-base")

def extractive_summary(text, sentence_count=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join([str(sentence) for sentence in summary])

def summarise_script(tokenized_scenes, climax_position_percentage):
    # Flatten the tokenized scenes into a single list of tokens up to the climax position
    flat_tokens = [token for scene in tokenized_scenes for token in scene]
    climax_token_count = int((climax_position_percentage / 100) * len(flat_tokens))

    tokens_up_to_climax = flat_tokens[:climax_token_count]

    # Convert the tokens back to text
    text_up_to_climax = " ".join(tokens_up_to_climax)

    # Extractive summary
    extractive_text = extractive_summary(text_up_to_climax, sentence_count=5)

    # Refine the prompt to guide summarization
    input_text = (
        "summarize the following script as a TV episode description, capturing the essence of the story: "
        + extractive_text
    )

    # Tokenize the input and generate the summary
    inputs = tokenizer(input_text, return_tensors="pt", max_length=7500, truncation=True)
    summary_ids = model.generate(inputs.input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary