from chatbot_together import analyze_matches as llm_analyzer

def analyze_matches(pickle_path, students):
    return llm_analyzer(pickle_path, students)
