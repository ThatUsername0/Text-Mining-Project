import openai
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

openai.api_key = ""

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

    # Call OpenAI API for summarization
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_text}
        ],
        max_tokens=150,
        temperature=0.7,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    summary = response.choices[0].message["content"].strip()
    return summary