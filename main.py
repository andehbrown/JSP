import platform # for getting information about the platform to determine hardware setup
import yaml # to read yml config file
from data import Data # data store module
from retriever import Retriever # retriever module
from generator import Generator # generator module
from evaluator import Evaluator # evaluator module
from timeit import default_timer as timer # for timing execution
import logging # to turn off annoying logs to terminal
from sentence_transformers import SentenceTransformer # for encoding chunks of text

logging.disable(logging.INFO)
logging.disable(logging.WARNING)
logging.disable(logging.ERROR)

# Load program variables from config.yml
config = yaml.safe_load(open('config.yml'))

# Load the vector encoder named in the config file. Enable CUDA hardware acceleration on Linux.
if platform.system() == 'Linux':
    VECTOR_ENCODER = SentenceTransformer(config['VECTOR_ENCODER'], device='cuda')
else:
    VECTOR_ENCODER = SentenceTransformer(config['VECTOR_ENCODER'])

# Instantiate modules
data = Data(config, VECTOR_ENCODER)
retriever = Retriever(config, VECTOR_ENCODER, data)
generator = Generator(config)

# Get user input, retrieve relevant data and send to LLM.
mode = input("Enter 'e' to evaluate the model or any other key to start interacting with JSPs.\n\n")

match mode:

    # Evaluation mode - evaluates the model against RAGAs metrics on sample questions in data/questions.csv    
    case 'e':

        evaluator = Evaluator(config, retriever, generator)
        evaluator.evaluate_with_ragas()
        print('Evaluation complete. Exiting program.')
        
    # Normal mode. Ask questions via free text.
    case _:
        
        while True:

            prompt = input("Ask a question about MOD policy from within JSPs or hit enter to quit. Simple questions work best.\n\n")
            if prompt=='':
                break
            # Retrieve prompt-relevant texts from the reference corpus
            reference_string, references = retriever.get_nearest_neighbours(prompt)
            generator.print_references(references)
            start = timer()
            # Generate a response to the prompt based upon the retrieved texts
            response = generator.get_llm_response(reference_string, prompt)[1:-1]
            end = timer()
            duration = end - start

            print(f"Response generation took {duration:.0f} seconds. This response may not represent authoritative policy and should be checked for accuracy against the source documents.\n")
            print(f"{prompt}\n")
            print(f"{response}.\n")
    