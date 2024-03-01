# JSP question-answering agent

Additional requirements

- The libraries listed in requirements.txt
- A conda install with the 'faiss' library (conda install faiss-gpu or faiss-cpu according to your hardware/preferences)
- The model to be placed in models/: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf?download=true

Instructions for running the program

- Choose program parameters in config.yml
- Run main.py
- If the selected parameters have been used before the JSP text chunks and vectors will be loaded from storage; selecting previously unused parameters will initiate chunking and vector embedding for PDF and ODT files in the documents/.
- At the prompt:
  - Type 'e' and hit return to begin evaluation using the parameters in config.yml against the questions in eval/questions.csv
  - Hit return to begin interactive mode where you can ask questions based on the PDF and ODT files in the sub-folder documents.
