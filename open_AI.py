from openai import OpenAI
import os
import PyPDF2
import json

with open("job_data.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# print(data)  # data is now a Python list/dict depending on JSON structure

with open("dummy_links.json", "r", encoding="utf-8") as file:
    data_links = json.load(file)

# print(data_links)

def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def read_pdf_file(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def import_resume(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"No file found at {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        return read_text_file(file_path)
    elif ext == ".pdf":
        return read_pdf_file(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .txt or .pdf file.")

def main():
    client = OpenAI(
        api_key="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",
        base_url="https://integrate.api.nvidia.com/v1"
    )

    path = input("Enter path to your resume file (.txt or .pdf): ").strip()
    try:
        resume_text = import_resume(path)
        print("Resume imported successfully!\n")
        job_search = input("What role are you looking into?")

            # Compose messages: system with /think, user with combined resume + question
        messages = [
            {"role": "system", "content": "Only answer in 1 paragraph."},
            {"role": "user", "content": f"Iterate through the given job data here: '\n{data}'. Then, go through this resume here: '\n\nResume:\n{resume_text}\n\n. Then based on the clients resume, suggest skills to and links for courses to follow, based on the data here. Return the links themselves: \n{data_links}."}
        ]

        response = client.chat.completions.create(
            model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
            messages=messages,
            temperature=0.6,
            top_p=0.95,
            max_tokens=2048,
            stream=True,
        )

        full_response = ""
        for chunk in response:
            content = chunk.choices[0].delta.content or ""
            full_response += content

# Print the whole thing at once
        print("\nAnswer:\n")
        print(full_response)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
