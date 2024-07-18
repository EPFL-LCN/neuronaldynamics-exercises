import openai

def correct_code(code_snippet):
    openai.api_key = 'sk-...oXLb'
    
    response = openai.Completion.create(
      engine="text-davinci-003",
      prompt=f"Correct the following code:\n\n{code_snippet}\n\n",
      max_tokens=200,
      n=1,
      stop=None,
      temperature=0.5,
    )

    return response.choices[0].text.strip()

if __name__ == "__main__":
    code_snippet = """
def example():
print("Hello, world!")
    """
    corrected_code = correct_code(code_snippet)
    print("Corrected Code:\n", corrected_code)
