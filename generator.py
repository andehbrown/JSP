import platform # for getting information about the platform to determine hardware setup
from llama_cpp import Llama # wrapper for utilising the LLM

class Generator:
    '''
    Contains the configuration data from an LLM and the methods associated with generating an LLM response
    and displaying references
    '''
               
    def __init__(self, config):
        
        self.config = config
        self.MODEL = config['MODEL']
        self.TEMPERATURE = config['TEMPERATURE']
        self.TOP_P = config['TOP_P']
        self.MIN_P = config['MIN_P']
        self.CONTEXT_LENGTH = config['CONTEXT_LENGTH']
    

    def get_llm_response(self, reference_string: str="", prompt: str="") -> str:
        '''
        Generates a response to a given prompt with reference to any context provided in the reference string
        
        Parameters
        ----------
        relevant JSP data: str
            used as context
        user prompt: str
            normally the question from the user
        
        Returns
        -------
        str
            LLM response to prompt using data        
        '''

        if reference_string == "":
            guidance = "Respond with 'No answer to this question is available within the reference text'"
            question = "Respond with 'No answer to this question is available within the reference text'"
        else:
            guidance = f'You will be provided with reference text delimited by triple quotes and a question. Your task is to answer the question using only the provided reference text and to cite the paragraph(s) used to answer the question. If the reference text does not contain the information needed to answer this question then simply write: "No answer to this question has been found in the provided reference text." If an answer to the question is provided, it must be annotated with a citation.'
            question = f'"""{reference_string}""" Question: {prompt}'

        if platform.system()=='Linux':
            llm_chat = Llama(model_path=self.MODEL,
                            n_gpu_layers=32, # use CUDA for processing if GPU available - only on Linux
                            verbose=False,
                            n_batch=512,
                            n_predict=-1,
                            n_keep=1,
                            n_ctx=self.CONTEXT_LENGTH)
        else:
            llm_chat = Llama(model_path=self.MODEL,
                            verbose=False,
                            n_batch=512,
                            n_predict=-1,
                            n_keep=1,
                            n_ctx=self.CONTEXT_LENGTH)

        output = llm_chat.create_chat_completion(messages = [{"role": "system",
                                                            "content": guidance},
                                                            {"role": "user",
                                                            "content": question}],
                                                            temperature=self.TEMPERATURE,
                                                            top_p=self.TOP_P,
                                                            min_p=self.MIN_P)

        return output['choices'][0]['message']['content']
    
           
    def print_references(self, references: list) -> None:
        '''
        Pretty prints the references provided as argument
        
        Parameters
        ----------
        references: list
            containing: filename: str, page numbers from filename: list, % total similarity: float
            
        Returns
        -------
        None. Prints to terminal.
        '''
        
        # references = self._group_references(references)

        if len(references)==0:
            print("No relevant information was found in the available document set. Any response does not necessarily represent MOD policy.")

        else:
            # get needed whitespace after each filename
            longest_filename = 0
            for reference in references:
                longest_filename = max(longest_filename, len(reference[0]))

            spacer=" "*(longest_filename - len("File name"))
            
            # print the references
            print("\nThe following references influenced the response. Contains public sector information\
            licensed under the Open Government Licence v3.0. You can read the terms of the licence here:\
            https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/\n")
            print("File name",spacer,"Score Page numbers")
            for reference in references:
                filename = reference[0]
                pagelist = reference[1]
                distance = reference[2]
                spacer1 = " " * (longest_filename - len(filename) + 1)
                if distance <= 9.994:
                    spacer2="  "
                else:
                    spacer2=" "
                print(f"{filename}{spacer1}{distance:.2f}%{spacer2}{pagelist}")
            print()
        

    def _group_references(self, references: list) -> list:
        '''
        From the provided reference list, any filenames that have multiple pages referenced are put onto one line
        and their similarities summed
        
        Parameters
        ----------
        references: list
            containing: filename: str, page numbers from filename: list, % total similarity: float
            Filenames appear once per page in the references with a single similarity for each page
        
        Returns
        -------
        references: list
            containing: filename: str, page numbers from filename: list, % total similarity: float
            Filenames appear only once with pages from each contained within a list and the total of each page's
            similarity
        
        '''

        # create list for all page numbers in each unique filename and append to new list
        jsp_set = set([i[0] for i in references])
        grouped_references = []

        for filename in jsp_set:
            pages=[]
            distance=0
            for jsp in references:
                if filename==jsp[0]:
                    pages.append(jsp[1])
                    distance += jsp[2]
            pages = sorted(pages)
            grouped_references.append([filename, pages, distance])
            
        # sort by descending percentage contribution to answer
        grouped_references = sorted(grouped_references,key=lambda l:l[2], reverse=True)

        return grouped_references
    