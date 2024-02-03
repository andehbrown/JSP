import os
import fitz  # PyMuPDF is a PDF loader
from odf import text as odftxt, teletype  # Handling open document format files
from odf.opendocument import load  # Handling open document format files
import pandas as pd  # Pandas will handle tables in the PDF files and storage of vectors
import numpy as np # for managing arrays
import re  # regex for cleaning text
import faiss # for vector indexing and comparison search
from langchain.text_splitter import RecursiveCharacterTextSplitter # for splitting text into chunks

class Data:
    '''
    Contains reference data for RAG and the methods required to extract this from PDF and ODT files in DOCUMENTS
    '''
    
    def __init__(self, config, vector_encoder):
        
        self.config = config
    
        self.DOCUMENTS = config['DOCUMENTS']
        self.MODEL = config['MODEL']
        self.TEMPERATURE = config['TEMPERATURE']
        self.TOP_P = config['TOP_P']
        self.NUM_CHUNKS = config['NUM_CHUNKS']
        self.CHUNK_SIZE = config['CHUNK_SIZE']
        self.CHUNK_OVERLAP = self.CHUNK_SIZE / 10
        self.VECTOR_ENCODER = vector_encoder
        
        self.VECTORS_PKL = f'data/text_vectors-chunks={str(self.CHUNK_SIZE)}.pkl'
        self.CORPUS_PKL = f'data/text_corpus-chunks={str(self.CHUNK_SIZE)}.pkl'
        self.CHUNKS_PKL = f'data/text_chunks-chunks={str(self.CHUNK_SIZE)}.pkl'
        self.VECTORS_CSV = f'data/text_vectors-chunks={str(self.CHUNK_SIZE)}.csv'
        self.CORPUS_CSV = f'data/text_corpus-chunks={str(self.CHUNK_SIZE)}.csv'
        self.CHUNKS_CSV = f'data/text_chunks-chunks={str(self.CHUNK_SIZE)}.csv'
        
        # Set page margins to ignore headers/footers/printed page numbers/etc.
        self.x0 = 10   # left margin size
        self.x1 = 10   # right margin size
        self.y0 = 50  # top margin size
        self.y1 = 70  # bottom margin size
        
        self.jsp_corpus = None
        self.get_jsps()
        self.jsp_vectors = None
        self.get_vectors()
        self.jsp_index = None
        self.get_index()
        
        
    def clean_text(self, text: str) -> str:
        '''
        Prepares text for use by the LLM, removing special characters to improve reading comprehension. 
        Returns empty string if text=None.
        
        Parameters
        ----------
        text: str
            The text to be cleaned.
        
        Returns
        text: str
            Cleaned text
        
        '''
        if text == None:
            text = ""
        
        text = re.sub(r'\s{2,}', ' ', text)  # replaces multiple spaces with a single space
        text = re.sub(r'(\.\s){2,}', '.', text)  # replaces multiple periods followed by a space 
        text = re.sub(r'\.{2,}', '.', text)  # replaces multiple periods with single period
        text = re.sub(r'(\n\s*)+\n+', '   ', text)  # replaces multipline line breaks with a triple space
        text = text.replace('\n', '  ')  # replaces line breaks with a double space
        
        text = text.replace('‚Äô', "'")  # replaces poorly parsed 'left single quote'
        text = text.replace('‚Äò', " ")  # replaces poorly parsed 'right single quote'
        text = text.replace('ÔÉº', "-")  # replaces poorly parsed bullet
        text = text.replace('ÔÇ®', "-")  # replaces poorly parsed bullet
        text = text.replace('ÔÇ∑', "-")  # replaces poorly parsed bullet
        text = text.replace('ÔÇ£', "-")  # replaces poorly parsed bullet
        text = text.replace('ÔÄµ', "-")  # replaces poorly parsed bullet
        text = text.replace('¬', "-")    # replaces poorly parsed bullet
        text = text.replace('â€¢', "-")  # replaces poorly parsed bullet
        text = text.replace('™', "")     # replaces poorly parsed whitespace
        text = text.replace('â', "")     # replaces poorly parsed whitespace
        text = text.replace('Ä', "")     # replaces poorly parsed whitespace
        text = text.replace('Â', "")     # replaces poorly parsed whitespace
        text = text.replace('ì', "")     # replaces poorly parsed whitespace
        text = text.replace('ï', "")     # replaces poorly parsed whitespace
        text = text.replace('ù', "")     # replaces poorly parsed whitespace
        text = text.replace('µ', "")     # replaces poorly parsed whitespace
        text = text.replace('¶', "")     # replaces poorly parsed whitespace
        text = text.replace('€', "")     # replaces poorly parsed whitespace

        return text


    def get_odt_text(self, filename: str, path: str) -> str:
        '''
        Extracts plain text from ODT files
        
        Parameters
        ----------
        filename: str
            The name of the ODT file
        path: str
            The file's location      

        Returns
        -------
        The parsed text from the ODT file: str        
        
        '''
        odt = load(path + filename)
        paragraphs = odt.getElementsByType(odftxt.P)
        text = ""

        for i in range(len((paragraphs))):
            text = text + " " + teletype.extractText(paragraphs[i]) + " "

        return text


    def get_pdf_text(self, filename: str, path: str) -> tuple[str, list]:
        '''
        Extracts plain text from PDF files and returns it along with a 'page finder'
        
        Parameters
        ----------
        filename: str
            The name of the ODT file
        path: str
            The file's location      

        Returns
        -------
        text: str
            Contains the parsed text from the PDF file
        page_finder: list
            contains a list of the character numbers, counted from the start of the document, at the start of each page.
        
        '''        
        # Open the PDF
        jsp = fitz.open(path + filename)
        num_pages = len(jsp)
        text = ""
        page_finder = []
        
        # For each page in the PDF, extract and clean text, excluding the margins to remove page numbers, footers etc.
        
        for page_num in range(num_pages):
            page = jsp[page_num]
            cropbox = fitz.Rect(self.x1, self.y0, page.rect.width - self.x1, page.rect.height - self.y1)
            page_text = page.get_textbox(cropbox)
            page_text = self.clean_text(page_text)
            page_start_word = len(text)
            page_finder.append(page_start_word)
            text = text + " " + page_text

        return text, page_finder


    def get_jsps(self) -> None:
        '''
        Retrieves text from PDF and ODT files in DOCUMENTS and sets it to a dataframe, jsp_corpus_df
        The data frame contains the filename, the text in the file, and (for PDF files only) the index of the
        first character on each page
        '''        
        # Check for saved copy of JSP data
        if os.path.isfile(self.CORPUS_PKL):

            self.jsp_corpus = pd.read_pickle(self.CORPUS_PKL)
            return
        
        # Open the folder
        folder = os.listdir(self.DOCUMENTS)
        jsp_corpus = []

        # for each document add the filename, text contents, and page index to a list
        success = 0
        failed_files = []

        for filename in folder:

            status = "Loading {pc_complete:.2f}% complete. ".format(pc_complete = success / len(folder) * 100)
            print(status, f"Currently processing {filename}.                  ", end="\r")

            if filename.endswith('.pdf'):
                text, page_finder = self.get_pdf_text(filename, self.DOCUMENTS)
                jsp_corpus.append([filename, text, page_finder])
                success += 1
            elif filename.endswith('.odt'):
                text = self.get_odt_text(filename, self.DOCUMENTS)
                jsp_corpus.append([filename, text, []])
                success += 1
            else:
                failed_files.append(filename)
                continue

        jsp_corpus_df = pd.DataFrame(jsp_corpus, columns=['filename', 'text', 'page_finder'])
        jsp_corpus_df.to_pickle(self.CORPUS_PKL)

        print(f"Loading complete. Successfully loaded {success} of {len(folder)} files in {self.DOCUMENTS}.\n\nFailed to load: {failed_files}\n")
        self.jsp_corpus = jsp_corpus_df

    
    def get_chunks(self) -> pd.DataFrame:
        '''
        Breaks a string into chunks of target max length CHUNK_SIZE and saves it in a dataframe and to file.
        The text is split recursively by (in descending order of priority):
        1. Numbered lists (e.g. 1. 2. 3.)
        2. Lettered lists (e.g. a. b. c.)
        3. Double whitespace (cleaned from single newline to remove the \n\r character which can confuse LLM comprehension)
        4. Decimal (end of sentence)
        Until smaller than the target CHUNK_SIZE. If a split has to occur on a different character the chunks overlap by CHUNK_OVERLAP.
        
        jsp_chunks_df contains two columns: filename: str, text_chunk: str
        '''        
        jsp_chunks = []
        numbered_list_regex = re.compile(r'\s{2,3}[0-9]{1,3}\.')
        lettered_list_regex = re.compile(r'\s{2,3}[a-z]{1}\.')
        
        splitter = RecursiveCharacterTextSplitter(separators=[numbered_list_regex,lettered_list_regex,'\s{2}','\.'],
                                                is_separator_regex=True,
                                                chunk_size=self.CHUNK_SIZE,
                                                chunk_overlap=self.CHUNK_OVERLAP,
                                                keep_separator=True,
                                                strip_whitespace=True)

        for i, jsp in self.jsp_corpus.iterrows():
            chunks = splitter.split_text(jsp['text'])
            for chunk in chunks:
                jsp_chunks.append([jsp['filename'], chunk])

        jsp_chunks_df = pd.DataFrame(jsp_chunks, columns=['filename', 'text_chunk'])
        jsp_chunks_df.to_csv(self.CHUNKS_CSV)
        jsp_chunks_df.to_pickle(self.CHUNKS_PKL)

        return jsp_chunks_df


    def get_vectors(self) -> None:
        '''
        Creates vector embeddings from chunks of text in jsp_chunks which captures the semantic meaning of the chunk.
        Saves the vectors in a dataframe, jsp_vectors, containing: filename: str, text_chunk: str, vector: list of 1024 floats
        '''
        # if a vectorstore file is available, load rather than build
        if os.path.isfile(self.VECTORS_PKL):

            self.jsp_vectors = pd.read_pickle(self.VECTORS_PKL)
            return

        else:
            # for each row in the input df, embed the text chunk and add to a new df
            jsp_chunks = self.get_chunks()
            vectors = []
            total = len(jsp_chunks)

            for i, chunk in jsp_chunks.iterrows():
                vector = self.VECTOR_ENCODER.encode(chunk['text_chunk'], show_progress_bar=False, normalize_embeddings=True)
                vectors.append([chunk['filename'], chunk['text_chunk'], vector])
                print(f"Embedding text chunk {i+1} of {total} into a vector", end='\r')

            vectors_df = pd.DataFrame(vectors, columns=['filename', 'text_chunk', 'vector'])

            vectors_df.to_pickle(self.VECTORS_PKL)

            self.jsp_vectors = vectors_df


    def get_index(self):
        '''
        Creates a FAISS index of the vectors in jsp_vectors for fast comparison and retrieval. Saved as a FAISS index object.
        '''
        # create an empty np array of the appropriate size

        vector_dimension = len(self.jsp_vectors['vector'].iloc[0])
        vector_count = len(self.jsp_vectors.index)

        index_db = np.zeros(shape=(vector_count, vector_dimension), dtype='float32')

        # add an index and a vector to each row of the np array
        for i, vector in self.jsp_vectors.iterrows():
            index_db[i] = self.jsp_vectors['vector'][i]

        # convert the np array to an index
        jsp_index = faiss.IndexFlatL2(vector_dimension)
        jsp_index.add(index_db)

        self.jsp_index = jsp_index
    