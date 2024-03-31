import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_text(text):
    # Convert text to lowercase and remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    return text

def calculate_cosine_similarity(paragraph1, paragraph2):
    # Preprocess the paragraphs
    paragraph1 = preprocess_text(paragraph1)
    paragraph2 = preprocess_text(paragraph2)

    # Create a CountVectorizer to convert paragraphs to word frequency vectors
    vectorizer = CountVectorizer()
    vectorizer.fit([paragraph1, paragraph2])
    vector_paragraph1 = vectorizer.transform([paragraph1])
    vector_paragraph2 = vectorizer.transform([paragraph2])

    # Calculate cosine similarity between the vectors
    similarity_score = cosine_similarity(vector_paragraph1, vector_paragraph2)[0][0]
    return similarity_score


def calculate_stylistic_similarity(paragraph1, paragraph2):
    # calculate the similarity based on the ratio of sentence lengths in the paragraphs

    # Split paragraphs into sentences
    sentences1 = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', paragraph1)
    sentences2 = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', paragraph2)
    
    # Calculate average sentence length for each paragraph
    avg_sentence_length1 = sum(len(sentence.split()) for sentence in sentences1) / len(sentences1)
    avg_sentence_length2 = sum(len(sentence.split()) for sentence in sentences2) / len(sentences2)
    
    # Compute similarity score based on the ratio of average sentence lengths
    similarity_score = min(avg_sentence_length1, avg_sentence_length2) / max(avg_sentence_length1, avg_sentence_length2)
    
    return similarity_score
  
def calculate_grammar_similarity(paragraph1, paragraph2):
    # Placeholder function for grammar-based similarity
    # analyzing grammatical structures, sentence complexity, etc.
    # return similarity score
    return np.random.rand()

def calculate_error_similarity(paragraph1, paragraph2):
    # Placeholder function for error/mistake-based similarity
    # detecting spelling errors, grammatical mistakes, etc.
    # return similarity score
    return np.random.rand()

def calculate_grammar_type_similarity(paragraph1, paragraph2):
    # Placeholder function for word grammar type-based similarity
    # analyzing word categories (nouns, verbs, etc.)
    # similarity score
    return np.random.rand()

def main(paragraph1, paragraph2):
    # Calculate similarity scores for each method
    cosine_similarity_score = calculate_cosine_similarity(paragraph1, paragraph2)
    stylistic_similarity_score = calculate_stylistic_similarity(paragraph1, paragraph2)
    grammar_similarity_score = calculate_grammar_similarity(paragraph1, paragraph2)
    error_similarity_score = calculate_error_similarity(paragraph1, paragraph2)
    grammar_type_similarity_score = calculate_grammar_type_similarity(paragraph1, paragraph2)

    # Print similarity scores for each method
    print("Cosine Similarity Score:", cosine_similarity_score)
    print("Stylistic Similarity Score:", stylistic_similarity_score)
    print("Grammar Similarity Score:", grammar_similarity_score)
    print("Error Similarity Score:", error_similarity_score)
    print("Grammar Type Similarity Score:", grammar_type_similarity_score)

    # Write similarity scores to a text file
    with open("similarity_scores.txt", "w") as file:
        file.write("Cosine Similarity Score: " + str(cosine_similarity_score) + "\n")
        file.write("Stylistic Similarity Score: " + str(stylistic_similarity_score) + "\n")
        file.write("Grammar Similarity Score: " + str(grammar_similarity_score) + "\n")
        file.write("Error Similarity Score: " + str(error_similarity_score) + "\n")
        file.write("Grammar Type Similarity Score: " + str(grammar_type_similarity_score) + "\n")

    # Determine overall similarity based on a threshold
    overall_similarity_score = (cosine_similarity_score + stylistic_similarity_score + grammar_similarity_score + error_similarity_score + grammar_type_similarity_score) / 5
    if overall_similarity_score > 0.5:
        print("Likely written by the same author.")
    else:
        print("Likely written by different authors.")

if __name__ == "__main__":
    paragraph1 = "This is the first paragraph written by the author."
    paragraph2 = "This is another paragraph by the same author."
    main(paragraph1, paragraph2)
