from odf.opendocument import load  # Handling open document format files
import numpy as np # for managing arrays

class Retriever:
    '''
    Contains the methods needed to retrieve chunks of text based on similarity to a given prompt
    '''
    def __init__(self, config, vector_encoder, data):
        
        self.config = config        
        self.NUM_CHUNKS = config['NUM_CHUNKS']
        self.VECTOR_ENCODER = vector_encoder
        self.jsp_corpus = data.jsp_corpus
        self.jsp_vectors = data.jsp_vectors
        self.jsp_index = data.jsp_index
        

    def get_nearest_neighbours(self, text: str) -> tuple[str, list]:
        '''
        Returns the NUM_CHUNKS chunks of text that are most similar to the input text based on cosine similarity between
        vector embeddings of the input text and chunks in an index.
        
        Parameters
        ----------
        text: str
            Input text upon which to retrieve the most similar text chunks.
        
        Returns
        -------       
        reference_string: str
            All of the most similar 
        references: list
            A list of references from which the chunks are taken, containing:
            filename: str, page number of text in filename: int, % similarity to input text: float, chunk of text: str
        '''
        num_nearest_neighbours = self.NUM_CHUNKS
        # Embed input text as a vector
        vector = np.asarray([self.VECTOR_ENCODER.encode(text, show_progress_bar=False)])

        # Seach index for vector and return number of nearest neighbour chunks
        euclidean_distances, nearest_neighbours = self.jsp_index.search(vector, num_nearest_neighbours)
        reference_string = ""
        references = []
                
        euclidean_distances, nearest_neighbours = self._refine_nearest_neighbours(euclidean_distances, nearest_neighbours)

        if len(nearest_neighbours)==0:
            # reference_string = 'Do not answer this question as there are is no relevant information in the reference text. You must respond only with: "No answer to this question has been found in the provided reference text.". No other answer is to be provided.'
            reference_string = ""
        else:
        # Search for index location of text chunk in document and return page number
            
            for i, nearest_neighbour in enumerate(nearest_neighbours[0]):

                filename = self.jsp_vectors['filename'].iloc[nearest_neighbour]
                text_chunk = self.jsp_vectors['text_chunk'].iloc[nearest_neighbour]
                distance = euclidean_distances[0][i]

                page_finder = self.jsp_corpus[self.jsp_corpus['filename']==filename]['page_finder'].values[0]
                fulltext = self.jsp_corpus[self.jsp_corpus['filename']==filename]['text'].values[0]

                # Search for the index of the first 20 characters of text chunk in the full text of the JSP
                chunk_index = fulltext.find(text_chunk[0:20])
                page_num = 0
                for j, page_start in enumerate(page_finder):
                    if page_start >= chunk_index or j==len(page_finder)-1:
                        page_num = j + 1
                        break

                # Explanatory text for LLM to say where the reference comes from
                reference_string += f'This passage is from {filename}, page {page_num}: "{text_chunk}"  \n'
                # references.append([filename, page_num, distance/total_distance*100, text_chunk])
                references.append([filename, page_num, distance, text_chunk])

        return reference_string, references
    
    def _refine_nearest_neighbours(self, euclidean_distances: np.array, nearest_neighbours: np.array, max_euclidean_distance: float=0.85) -> tuple[np.array, np.array]:
        '''
        Reduces the np arrays of euclidean distances and nearest neighbours to remove entries from both with a euclidean distance
        greater than max_euclidean_distance. Further removes those over 1.1x the minimum to remove outliers.
        
        Parameters
        ----------
        euclidean_distances: np.array
            As returned by the FAISS search method - the distances of the closest matches to the search term
        nearest_neighbours: np.array
            As returned by the FAISS search method - the index of the chunks that are the closest match to the search term
        max_euclidean_distance: float
            The max value in the returned list. Default 0.85.
        
        Returns
        -------
        refined_euclidean_distances: np.array
            All of the euclidean distances less those over max_euclidean_distance
        refined_nearest_neighbours: np.array
            All of the nearest_neigbour text chunks with the same reductions applied
        '''       
        refined_euclidean_distances = []
        refined_nearest_neighbours = []
        
        for i, distance in np.ndenumerate(euclidean_distances[0]):
            if distance < max_euclidean_distance:
                refined_euclidean_distances.append(euclidean_distances[0][i])
                refined_nearest_neighbours.append(nearest_neighbours[0][i])
                
        # convert to np.array prior to return to keep data structure the same as original
        refined_euclidean_distances = np.array([refined_euclidean_distances])
        refined_nearest_neighbours = np.array([refined_nearest_neighbours])
        
        return refined_euclidean_distances, refined_nearest_neighbours
    