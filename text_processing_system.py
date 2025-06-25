
import re
import requests
import numpy as np
import math
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple, Optional, Any
from enum import Enum
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Try to import NLTK
try:
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
        from nltk.tokenize import word_tokenize
        from nltk.tag import pos_tag
        from nltk.stem import WordNetLemmatizer
        from nltk.corpus import stopwords, wordnet
        NLTK_AVAILABLE = True
    except LookupError:
        print("Download NLTK data...")
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('stopwords', quiet=True)
            from nltk.tokenize import word_tokenize
            from nltk.tag import pos_tag
            from nltk.stem import WordNetLemmatizer
            from nltk.corpus import stopwords, wordnet
            NLTK_AVAILABLE = True
        except:
            print("The NLTK data download failed. Use a simplified implementation....")
            NLTK_AVAILABLE = False
except ImportError:
    print("NLTK is not installed, use a simplified implementation...")
    NLTK_AVAILABLE = False

# Try to import BeautifulSoup
try:
    import bs4
    BS4_AVAILABLE = True
except ImportError:
    print("BeautifulSoup4 is not installed, and the web scraping function will be unavailable....")
    BS4_AVAILABLE = False

# =============================================================================
# Module1: (TextCollector)
# =============================================================================

class TextCollector:
    """
    Main methods
    - collect_texts(): Main collection methods
    - collect_from_gutenberg(): Collect texts from Project Gutenberg
    - save_texts(): Save the collected text to a file
    """

    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.collected_texts = []
    
    def collect_from_gutenberg(self) -> List[str]:
        """
        1. Check if BeautifulSoup4 is available
        2. Visit the specified Project Gutenberg page
        3. Extract the paragraph text of a specific class
        4. Return the list of extracted text
        """
        # check bs4
        if not BS4_AVAILABLE:
            print("BeautifulSoup4 is not available")
            return []

        print("Collecting texts from Project Gutenberg...")
        url = "https://www.gutenberg.org/cache/epub/76314/pg76314-images.html"

        try:
            # Send an HTTP request to obtain the page content
            print(f"visiting : {url}")
            response = requests.get(url, headers=self.headers, timeout=15)

            if response.status_code != 200:
                print(f"HTTP request failed, status code: {response.status_code}")
                return []

            print(f"Successfully obtained the page content and size: {len(response.text)} ")
            soup = bs4.BeautifulSoup(response.text, 'html.parser')

            # Extract the paragraph text of the specified class
            texts = []

            # Extract the first paragraph (drop-capa0_0_6 c008)
            print("🔍 Searching for the first paragraph...")
            text1_element = soup.find('p', class_="drop-capa0_0_6 c008")
            if text1_element:
                text1 = text1_element.text.strip()
                if text1 and len(text1.split()) >= 10:  # 至少10个单词
                    texts.append(text1)
                    print(f"✅ Successfully extracted the text of the 1st paragraph: {len(text1.split())} words")
                    print(f"📝 Content preview: {text1[:100]}...")
                else:
                    print("⚠️ The first paragraph of text is too short, skip it.")
            else:
                print("❌ The first paragraph was not found (class='drop-capa0_0_6 c008')")

            # Extract the second paragraph (c009)
            print("🔍 Searching for the second paragraph...")
            text2_element = soup.find('p', class_="c009")
            if text2_element:
                text2 = text2_element.text.strip()
                if text2 and len(text2.split()) >= 10:  # 至少10个单词
                    texts.append(text2)
                    print(f"✅ Successfully extracted the text of the second paragraph: {len(text2.split())} 个单词")
                    print(f"📝 Content preview: {text2[:100]}...")
                else:
                    print("⚠️ The text in the second paragraph is too short, so skip it.")
            else:
                print("❌ The second paragraph was not found (class='c009')")

            # Try to extract more paragraphs with the same class
            print("🔍 Searching for more paragraphs...")
            additional_paragraphs = soup.find_all('p', class_="c009")
            print(f"📊 found {len(additional_paragraphs)}  'c009' class paragraphs")

            for i, para in enumerate(additional_paragraphs[1:6]):  # Take at most 5 more paragraphs
                text = para.text.strip()
                if text and len(text.split()) >= 10:
                    texts.append(text)
                    print(f"✅ Successfully extract the text of the {i+3}th paragraph: {len(text.split())} words")
                    print(f"📝 Content preview: {text[:100]}...")

            if texts:
                print(f"🎉 A total of collected {len(texts)} paragraphs of text")
                return texts
            else:
                print("❌ No valid text was extracted")
                # Try to extract any paragraphs as alternatives
                print("🔄 Try to extract any available paragraphs...")
                all_paragraphs = soup.find_all('p')
                print(f"📊 There are in total {len(all_paragraphs)} paragraphs on the page")

                backup_texts = []
                for i, para in enumerate(all_paragraphs[:10]):  # Take at most the first 10 paragraphs
                    text = para.text.strip()
                    if text and len(text.split()) >= 20:  # Raise the standard to 20 words
                        backup_texts.append(text)
                        print(f"✅ Alternative text{len(backup_texts)}: {len(text.split())} words")
                        if len(backup_texts) >= 3:  # At most 3 alternative texts
                            break

                return backup_texts

        except requests.exceptions.Timeout:
            print("❌ The network request timed out. Please check your network connection.")
            return []
        except requests.exceptions.ConnectionError:
            print("❌ The network request timed out. Please check your network connection.")
            return []
        except Exception as e:
            print(f"❌ An error occurred while collecting text from Project Gutenberg. {e}")
            return []

    def get_fallback_texts(self) -> List[str]:
        """
        Provide alternative text (when the network is unavailable)
        """
        print("🔄 Use alternative text for demonstration...")
        return [
            """In the heart of London, where the Thames flows through centuries of history,
            stands the magnificent Tower Bridge. This iconic Victorian Gothic structure,
            completed in 1894, represents the perfect marriage of engineering prowess and
            architectural beauty. The bridge's twin towers, connected by high-level walkways
            and a central bascule mechanism, have witnessed the transformation of London
            from a bustling port city to a modern metropolis. Thousands of visitors daily
            marvel at its intricate stonework and the spectacular views it offers of the
            city skyline. The Tower Bridge Exhibition provides fascinating insights into
            the bridge's construction and operation, making it one of London's most beloved
            landmarks and a testament to human ingenuity.""",

            """The digital revolution has fundamentally transformed how we communicate, work,
            and live our daily lives. From the early days of personal computers to today's
            smartphones and artificial intelligence, technology continues to reshape society
            at an unprecedented pace. Social media platforms connect billions of people
            across the globe, enabling instant communication and information sharing.
            E-commerce has revolutionized retail, allowing consumers to purchase goods
            from anywhere in the world with a few clicks. Remote work, once a rare
            privilege, has become commonplace, changing traditional office dynamics.
            However, this digital transformation also brings challenges including privacy
            concerns, digital divide issues, and the need for continuous adaptation to
            new technologies.""",

            """Climate change represents one of the most pressing challenges of our time,
            requiring urgent global action and cooperation. Rising temperatures, melting
            ice caps, and extreme weather events are clear indicators of our planet's
            changing climate system. Scientists worldwide have reached consensus that
            human activities, particularly greenhouse gas emissions from fossil fuel
            consumption, are the primary drivers of current climate change. Governments,
            businesses, and individuals must work together to implement sustainable
            solutions including renewable energy adoption, energy efficiency improvements,
            and conservation efforts. The transition to a low-carbon economy presents
            both challenges and opportunities, requiring innovation, investment, and
            commitment from all sectors of society to ensure a sustainable future for
            generations to come."""
        ]

    def collect_texts(self, use_network: bool = True) -> List[str]:
        """
        The main methods of collecting text

        Parameters:
            use_network (bool): Whether to try network collection，default True

        Function:
        1. If use_network=True, try to collect text from the Project Gutenberg website
        2. If network collection fails or use_network=False, use the alternative text.
        """
        texts = []

        if use_network:
            # First, check the network connection.
            try:
                print("🔍 Check the network connection...")
                test_response = requests.get("https://httpbin.org/get", timeout=3)
                if test_response.status_code == 200:
                    print("✅ The network connection is normal. Try to collect text....")
                    texts = self.collect_from_gutenberg()
                else:
                    print("⚠️ There is an abnormal network connection. Use the alternate text.")
            except:
                print("⚠️ The network is unavailable. Use the alternate text.")
        else:
            print("📖 Directly use the alternate text mode")

        if not texts:
            print("🔄 Switch to alternate text mode")
            texts = self.get_fallback_texts()

        if not texts:
            print("❌ No text was obtained, and the system cannot continue.")
            return []

        self.collected_texts = texts
        print(f"✅ Successfully obtained {len(texts)} paragraphs")
        for i, text in enumerate(texts):
            word_count = len(text.split())
            print(f"📄 text {i+1}: {word_count} words")
        self.save_texts()

        return texts
    
    def save_texts(self, filename: str = "collected_texts.txt"):
        """Save the collected text to a file"""
        with open(filename, 'w', encoding='utf-8') as f:
            for i, text in enumerate(self.collected_texts):
                f.write(f"=== Text {i+1} ===\n")
                f.write(text.strip())
                f.write("\n\n")
        print(f"The text has been saved to {filename}")

# =============================================================================
# Module2: (CollocationFinder)
# =============================================================================

class TrieNode:
    """
    Prefix tree node class

    Property description：
    - children: Dictionary of child nodes, where the key is a word and the value is a TrieNode
    - is_end: Mark whether it is a phrase ending node
    - frequency: The frequency of occurrence of this phrase
    - phrases: Store the list of complete phrases ending with this node
    """
    def __init__(self):
        self.children = {}      # Dictionary of child nodes
        self.is_end = False     # Is it the end of the phrase
        self.frequency = 0      # Phrase frequency
        self.phrases = []       # Phrase list

class Trie:
    """
    Prefix tree (Trie) implementation class

    Function description：
    A prefix tree is a tree - shaped data structure used to efficiently store and retrieve a set of strings.
    In fixed collocation search, it is used to store and quickly match multi-word phrases.

    advantages：
    - High space efficiency: Share common prefixes
    - High search efficiency: O(m) time complexity, where m is the length of the phrase
    - Support prefix matching and pattern search
    """
    def __init__(self):
        """Initialize the prefix tree and create the root node"""
        self.root = TrieNode()

    def insert(self, phrase: str, frequency: int = 1):
        """
        Insert phrases into the prefix tree

        Parameters:
            phrase (str): Phrases to be inserted
            frequency (int): Phrase occurrence frequency

        Algorithm process:
        1. Start from the root node
        2. Traverse/create nodes level by level downwards according to words
        3. Mark phrase information at the end node
        """
        node = self.root
        words = phrase.lower().split()

        # Traverse word by word, create or move to child nodes
        for word in words:
            if word not in node.children:
                node.children[word] = TrieNode()
            node = node.children[word]

        # Mark the end of the phrase and record the information
        node.is_end = True
        node.frequency += frequency
        if phrase not in node.phrases:
            node.phrases.append(phrase)
    
    def search(self, phrase: str) -> bool:
        """Search whether the phrase exists"""
        node = self.root
        words = phrase.lower().split()
        
        for word in words:
            if word not in node.children:
                return False
            node = node.children[word]
        
        return node.is_end
    
    def get_frequency(self, phrase: str) -> int:
        """Get phrase frequency"""
        node = self.root
        words = phrase.lower().split()
        
        for word in words:
            if word not in node.children:
                return 0
            node = node.children[word]
        
        return node.frequency if node.is_end else 0

class SortedArray:
    """Implementation of sorted array"""
    def __init__(self):
        self.data = []
    
    def insert(self, phrase: str, frequency: int):
        """Insert phrases and frequencies, maintaining the sorting"""
        item = (frequency, phrase)
        left, right = 0, len(self.data)
        while left < right:
            mid = (left + right) // 2
            if self.data[mid][0] < frequency:
                left = mid + 1
            else:
                right = mid
        self.data.insert(left, item)
    
    def get_top_k(self, k: int) -> List[Tuple[int, str]]:
        """Get the k phrases with the highest frequency"""
        return self.data[-k:] if k <= len(self.data) else self.data[:]
    
    def search(self, phrase: str) -> int:
        """Search for phrase frequency"""
        for freq, p in self.data:
            if p == phrase:
                return freq
        return 0

class CollocationFinder:
    """
    Fixed Phrase Finder - Using Multiple Data Structures to Find Fixed Phrases in Text

    Function description：
    Collocations refer to combinations of words that frequently occur together,
    such as "artificial intelligence" and "climate change". In this category,
    three different data structures are used to store and search for collocations:

    1. Hash Table: Provides a lookup time complexity of O(1)
    2. Trie: Supports prefix matching and pattern search
    3. Sorted Array: Sort by frequency to facilitate the acquisition of high-frequency collocations

    Supported n-gram types:
    - Bigrams: combinations of two words
    - Trigrams: combinations of three words
    - Fourgrams: combinations of four words

    Main methods：
    - analyze_texts(): Analyze the text to extract fixed collocations
    - find_collocations_hash(): Use hash table for searching
    - find_collocations_trie(): Use a prefix tree for searching
    - get_top_collocations_sorted(): Obtain high-frequency collocations
    """
    def __init__(self):
        """
        Initialize the collocation finder

        Description of data structure：
        - hash_table: Hash table, storing the mapping of phrases -> frequencies
        - trie: Prefix tree, used for pattern matching
        - sorted_array: Sort the array, sort by frequency
        - texts: Original text list
        - bigrams/trigrams/fourgrams: Counters for storing 2/3/4 - yuan combinations respectively
        """
        self.hash_table = defaultdict(int)  # Hash table storage
        self.trie = Trie()                  # Prefix tree storage
        self.sorted_array = SortedArray()   # Sorted array storage
        self.texts = []                     # Original text
        self.bigrams = Counter()            # Binary combination counter
        self.trigrams = Counter()           # Three - element combination counter
        self.fourgrams = Counter()          # Quaternion combination counter

    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text, clean and standardize

        Parameters:
            text (str): Original text

        return:
            List[str]: Cleaned word list

        Processing steps:
        1. Convert to lowercase
        2. Remove punctuation marks, keeping only letters and spaces
        3. Split into words and filter out empty strings
        """
        # Remove punctuation marks and convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Split and filter out empty strings
        words = [word for word in text.split() if word.strip()]
        return words

    def extract_ngrams(self, words: List[str], n: int) -> List[str]:
        """
        Extract n-gram combinations from the word list

        parameters:
            words (List[str]): Word list
            n (int): n-gram（2=bigram, 3=trigram, 4=fourgram）

        return:
            List[str]: n-gram

        algorithm：
        Use the sliding window method with a window size of n, and slide from left to right to extract all possible n-grams.
        """
        if len(words) < n:
            return []
        # Extract n-grams using sliding window
        return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
    
    def analyze_texts(self, texts: List[str]):
        """Analyze the text and extract fixed collocations"""
        self.texts = texts
        print("Analyzing the fixed collocations in the text...")
        
        for text in texts:
            words = self.preprocess_text(text)
            
            bigrams = self.extract_ngrams(words, 2)
            trigrams = self.extract_ngrams(words, 3)
            fourgrams = self.extract_ngrams(words, 4)
            
            self.bigrams.update(bigrams)
            self.trigrams.update(trigrams)
            self.fourgrams.update(fourgrams)
        
        all_ngrams = dict(self.bigrams)
        all_ngrams.update(dict(self.trigrams))
        all_ngrams.update(dict(self.fourgrams))
        
        self._populate_data_structures(all_ngrams)
        
        print(f"Analysis completed！")
        print(f"found {len(self.bigrams)} bigrams")
        print(f"found {len(self.trigrams)} trigrams")
        print(f"found {len(self.fourgrams)} fourgrams")
        self.save_ngrams()
    
    def _populate_data_structures(self, ngrams: Dict[str, int]):
        """Populate the data structure"""
        filtered_ngrams = {phrase: freq for phrase, freq in ngrams.items() if freq >= 2}
        
        for phrase, frequency in filtered_ngrams.items():
            self.hash_table[phrase] = frequency
            self.trie.insert(phrase, frequency)
            self.sorted_array.insert(phrase, frequency)
    
    def find_collocations_hash(self, min_frequency: int = 2) -> List[Tuple[str, int]]:
        """Use a hash table to find fixed collocations"""
        print(f"\n=== Use a hash table to find fixed collocations（Minimum frequency: {min_frequency}）===")
        
        collocations = [(phrase, freq) for phrase, freq in self.hash_table.items() 
                       if freq >= min_frequency]
        collocations.sort(key=lambda x: x[1], reverse=True)
        
        print(f"found {len(collocations)} fixed collocations")
        self.save_collocations(collocations, "hash_table_collocations.txt")
        return collocations
    
    def find_collocations_trie(self, patterns: List[str]) -> List[Tuple[str, int]]:
        """Use a prefix tree to find fixed collocations"""
        print(f"\n=== Use a prefix tree to find fixed collocations ===")
        
        results = []
        for pattern in patterns:
            if self.trie.search(pattern):
                frequency = self.trie.get_frequency(pattern)
                results.append((pattern, frequency))
                print(f"Find the pattern '{pattern}': frequency {frequency}")
            else:
                print(f"Pattern not found '{pattern}'")
        self.save_collocations(results, "trie_collocations.txt")
        
        return results
    
    def get_top_collocations_sorted(self, k: int = 10) -> List[Tuple[int, str]]:
        """Use sorted arrays to obtain the k fixed collocations with the highest frequency"""
        print(f"\n=== Use a sorted array to obtain the top {k} high-frequency fixed collocations ===")
        
        top_collocations = self.sorted_array.get_top_k(k)
        top_collocations.reverse()
        
        for i, (freq, phrase) in enumerate(top_collocations, 1):
            print(f"{i}. '{phrase}': {freq} ")
        self.save_collocations(top_collocations, "top_collocations.txt")

        return top_collocations
    
    def search_collocation(self, phrase: str) -> Dict[str, int]:
        """Search for fixed collocations in all data structures"""
        print(f"\n=== Search for fixed collocations: '{phrase}' ===")
        
        results = {
            'hash_table': self.hash_table.get(phrase, 0),
            'trie': self.trie.get_frequency(phrase),
            'sorted_array': self.sorted_array.search(phrase)
        }
        
        for method, frequency in results.items():
            print(f"{method}: {frequency} ")
        with open("search_collocation_results.txt", 'w', encoding='utf-8') as f:
            for method, frequency in results.items():
                f.write(f"{method}: {frequency}\n")

        return results

    def save_ngrams(self, filename: str = "ngrams.txt"):
        """Save n-grams to a file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("Bigrams:\n")
            for bigram, freq in self.bigrams.items():
                f.write(f"{bigram}: {freq}\n")
            f.write("\nTrigrams:\n")
            for trigram, freq in self.trigrams.items():
                f.write(f"{trigram}: {freq}\n")
            f.write("\nFourgrams:\n")
            for fourgram, freq in self.fourgrams.items():
                f.write(f"{fourgram}: {freq}\n")
        print(f"The n-grams have been saved to {filename}")

    def save_collocations(self, collocations: List[Tuple[str, int]], filename: str):
        """Save collocations to a file"""
        with open(filename, 'w', encoding='utf-8') as f:
            for phrase, freq in collocations:
                f.write(f"{phrase}: {freq}\n")
        print(f"The collocations have been saved to {filename}")


# =============================================================================
# Module3: (BasicInvertedIndex)
# =============================================================================

class Document:
    """
    Function description：
    Encapsulate the basic information of the document and the pre-processed content,
    providing a standardized document representation for the inverted index.。

    Attributes：
    - doc_id:
    - content:
    - title:
    - words:
    - word_count:
    """
    def __init__(self, doc_id: int, content: str, title: str = ""):
        self.doc_id = doc_id
        self.content = content
        self.title = title
        self.words = self._preprocess_content()  # Preprocess to get a list of words
        self.word_count = len(self.words)        # Count the number of words

    def _preprocess_content(self) -> List[str]:
        # Remove punctuation marks and convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', self.content.lower())
        # Split and filter empty strings
        words = [word for word in text.split() if word.strip()]
        return words

class PostingList:
    """
    Inverted List Class - Store document information for a single term

    Function description:
    The inverted list is the core data structure of the inverted index, storing which documents each term appears in,
    as well as the specific position information of each term in each document.

    Attributes：
    - documents: Dictionary, where the key is the document ID and the value is a list of positions of the word in the document
    - document_frequency: The number of documents containing the word (used to calculate IDF)

    Time complexity：
    - Add documentation: O(1)
    - Get document collections: O(1)
    - Get location information: O(1)
    """
    def __init__(self):
        self.documents = {}           # Mapping of document ID -> list of positions
        self.document_frequency = 0   # Document frequency (the number of documents containing the word)

    def add_document(self, doc_id: int, positions: List[int]):
        """
        Add document and the location information of the word in the document

        parameters:
            doc_id (int):
            positions (List[int]):
        """
        # If it is a new document, increase the document frequency.
        if doc_id not in self.documents:
            self.document_frequency += 1
        # Store location information
        self.documents[doc_id] = positions

    def get_documents(self) -> Set[int]:
        """
        Get the set of all document IDs that contain this word

        return:
            Set[int]:
        """
        return set(self.documents.keys())

    def get_positions(self, doc_id: int) -> List[int]:
        """
        Get the list of positions of the word in a specific document
        """
        return self.documents.get(doc_id, [])

class BasicInvertedIndex:
    """
    Basic Inverted Index Class - The core data structure for implementing document retrieval

    Function description：
    The inverted index is a core data structure in information retrieval systems.
    It establishes a mapping relationship from terms to a list of documents that contain the terms.
    Contrary to the forward index (document -> term), the inverted index supports efficient keyword search.

    Data Structure：
    - index: Main index, mapping from term to PostingList
    - documents: Document storage, mapping from document ID to Document object
    - vocabulary: Glossary, containing all unique terms
    - total_documents: Total number of documents

    Supported search types:
    1. Word search: Find documents that contain a specific word.
    2. Multi-word AND search: Find documents that contain multiple words simultaneously.
    3. Multi-word OR search: Find documents that contain any of the words.
    4. Phrase search: Find documents that contain the complete phrase.
    5. Sorted search: Search results sorted based on TF-IDF scores.

    Time complexity：
    - Build an index: O(N*M)，N is the number of documents, and M is the average document length.
    - Word search: O(1)
    - Multi-word search: O(k*D)，k is the number of query terms, and D is the average length of the document list
    """
    def __init__(self):
        """
        Initialize the basic inverted index

        Data structure description：
        - index: Core inverted index, using defaultdict to automatically create PostingList
        - documents: Document storage dictionary
        - total_documents: Document counter
        - vocabulary: Glossary collection for quick vocabulary lookup
        """
        self.index = defaultdict(PostingList)
        self.documents = {}
        self.total_documents = 0
        self.vocabulary = set()

    def add_document(self, doc_id: int, content: str, title: str = ""):
        """
        Add a single document to the index

        Parameters:
            doc_id (int):
            content (str):
            title (str):

        Index building process:
        1. Create Document objects and preprocess the content.
        2. Count the positions of each word in the document.
        3. Update the vocabulary.
        4. Add the position information to the inverted list of the corresponding term.
        """
        # Create a document object
        document = Document(doc_id, content, title)
        self.documents[doc_id] = document
        self.total_documents += 1

        # Statistical word position information
        word_positions = defaultdict(list)

        # Traverse each word in the document and record its position.
        for position, word in enumerate(document.words):
            word_positions[word].append(position)
            self.vocabulary.add(word)

        # Add location information to the inverted index
        for word, positions in word_positions.items():
            self.index[word].add_document(doc_id, positions)

    def build_index(self, texts: List[str]):
        """
        Batch build index
        Function:
        Traverse all texts, create a document for each text and add it to the index.
        """
        print("Building the basic inverted index...")

        # Batch add documents
        for i, text in enumerate(texts):
            title = f"Document {i+1}"
            self.add_document(i, text, title)

        # Output index statistics
        print(f"Index building is completed！")
        print(f"Total number of documents: {self.total_documents}")
        print(f"Vocabulary size: {len(self.vocabulary)}")

        self.save_index_statistics()

    def search_single_word(self, word: str) -> Set[int]:
        """Search for a single word"""
        word = word.lower()
        if word in self.index:
            return self.index[word].get_documents()
        return set()

    def search_multiple_words_and(self, words: List[str]) -> Set[int]:
        """Multi - word AND search"""
        if not words:
            return set()

        result = self.search_single_word(words[0])

        for word in words[1:]:
            word_docs = self.search_single_word(word)
            result = result.intersection(word_docs)

            if not result:
                break

        return result

    def search_multiple_words_or(self, words: List[str]) -> Set[int]:
        """Multi - word OR search"""
        result = set()

        for word in words:
            word_docs = self.search_single_word(word)
            result = result.union(word_docs)

        return result

    def phrase_search(self, phrase: str) -> Set[int]:
        """Phrase search"""
        words = phrase.lower().split()
        if len(words) == 1:
            return self.search_single_word(words[0])

        candidate_docs = self.search_multiple_words_and(words)

        result = set()

        for doc_id in candidate_docs:
            if self._contains_phrase(doc_id, words):
                result.add(doc_id)

        return result

    def _contains_phrase(self, doc_id: int, words: List[str]) -> bool:
        """Check if the document contains the complete phrase"""
        if not words:
            return False

        first_word_positions = self.index[words[0]].get_positions(doc_id)

        for start_pos in first_word_positions:
            valid_phrase = True
            for i, word in enumerate(words[1:], 1):
                expected_pos = start_pos + i
                word_positions = self.index[word].get_positions(doc_id)

                if expected_pos not in word_positions:
                    valid_phrase = False
                    break

            if valid_phrase:
                return True

        return False

    def calculate_tf_idf(self, word: str, doc_id: int) -> float:
        """Calculate TF-IDF scores"""
        if word not in self.index or doc_id not in self.documents:
            return 0.0

        word_positions = self.index[word].get_positions(doc_id)
        tf = len(word_positions) / self.documents[doc_id].word_count

        df = self.index[word].document_frequency
        idf = math.log(self.total_documents / df) if df > 0 else 0

        return tf * idf

    def ranked_search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Sorting and Searching"""
        words = query.lower().split()
        doc_scores = defaultdict(float)

        for word in words:
            if word in self.index:
                for doc_id in self.index[word].get_documents():
                    doc_scores[doc_id] += self.calculate_tf_idf(word, doc_id)

        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        top_results = ranked_docs[:top_k]

        # Save ranked search results to a file
        self.save_ranked_search_results(top_results, query)

        return top_results

    def save_index_statistics(self, filename: str = "index_statistics.txt"):
        """Save index statistics to a file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Total number of documents: {self.total_documents}\n")
            f.write(f"Vocabulary size: {len(self.vocabulary)}\n")
        print(f"The index statistics have been saved to {filename}")

    def save_ranked_search_results(self, results: List[Tuple[int, float]], query: str,
                                   filename: str = "ranked_search_results.txt"):
        """Save ranked search results to a file"""
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(f"Query: {query}\n")
            for doc_id, score in results:
                title = self.documents[doc_id].title
                f.write(f"Document ID: {doc_id}, Title: {title}, Score: {score}\n")
            f.write("\n")
        print(f"The ranked search results for query '{query}' have been saved to {filename}")

# =============================================================================
# Module4: lexical analysis reverse index
# =============================================================================

class LexicalDocument(Document):
    """Document class with lexical analysis"""
    def __init__(self, doc_id: int, content: str, title: str = ""):
        try:
            if NLTK_AVAILABLE:
                self.lemmatizer = WordNetLemmatizer()
                self.stop_words = set(stopwords.words('english'))
            else:
                raise Exception("NLTK not available")
        except:
            self.lemmatizer = None
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}

        super().__init__(doc_id, content, title)
        self.pos_tags = []
        self.lemmas = []
        self._perform_lexical_analysis()

    def _preprocess_content(self) -> List[str]:
        """Preprocess the document content"""
        try:
            if NLTK_AVAILABLE:
                tokens = word_tokenize(self.content.lower())
                words = [token for token in tokens if token.isalpha() and len(token) > 1]
            else:
                raise Exception("NLTK not available")
        except:
            text = re.sub(r'[^\w\s]', ' ', self.content.lower())
            words = [word for word in text.split() if word.strip() and len(word) > 1]
        return words

    def _perform_lexical_analysis(self):
        """Perform lexical analysis"""
        try:
            if NLTK_AVAILABLE:
                self.pos_tags = pos_tag(self.words)
                self.lemmas = []
                for word, pos in self.pos_tags:
                    wordnet_pos = self._get_wordnet_pos(pos)
                    if wordnet_pos:
                        lemma = self.lemmatizer.lemmatize(word, wordnet_pos)
                    else:
                        lemma = self.lemmatizer.lemmatize(word)
                    self.lemmas.append(lemma)
            else:
                raise Exception("NLTK not available")
        except:
            self.pos_tags = [(word, 'NN') for word in self.words]
            self.lemmas = [self._simple_lemmatize(word) for word in self.words]

    def _simple_lemmatize(self, word: str) -> str:
        """Simplified lemmatization"""
        if word.endswith('ies'):
            return word[:-3] + 'y'
        elif word.endswith('s') and not word.endswith('ss'):
            return word[:-1]
        elif word.endswith('ed'):
            return word[:-2]
        elif word.endswith('ing'):
            return word[:-3]
        return word

    def _get_wordnet_pos(self, treebank_tag: str) -> Optional[str]:
        """Convert TreeBank POS tags to WordNet POS tags"""
        if not NLTK_AVAILABLE:
            return None
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None


class LexicalPostingList(PostingList):
    """Inverted list with lexical information"""
    def __init__(self):
        super().__init__()
        self.pos_info = defaultdict(list)
        self.lemma_info = defaultdict(list)

    def add_document_with_lexical_info(self, doc_id: int, positions: List[int],
                                     pos_tags: List[str], lemmas: List[str]):
        """Add documents with lexical information"""
        self.add_document(doc_id, positions)

        for i, pos in enumerate(positions):
            if i < len(pos_tags):
                self.pos_info[doc_id].append((pos, pos_tags[i]))
            if i < len(lemmas):
                self.lemma_info[doc_id].append((pos, lemmas[i]))


class LexicalInvertedIndex(BasicInvertedIndex):
    """Lexical analysis reverse index"""
    def __init__(self):
        super().__init__()
        self.lemma_index = defaultdict(LexicalPostingList)
        self.pos_index = defaultdict(LexicalPostingList)
        try:
            if NLTK_AVAILABLE:
                self.stop_words = set(stopwords.words('english'))
                self.lemmatizer = WordNetLemmatizer()
            else:
                raise Exception("NLTK not available")
        except:
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            self.lemmatizer = None

    def add_document(self, doc_id: int, content: str, title: str = ""):
        """Add documents to the lexical index"""
        document = LexicalDocument(doc_id, content, title)
        self.documents[doc_id] = document
        self.total_documents += 1

        word_positions = defaultdict(list)
        lemma_positions = defaultdict(list)
        pos_positions = defaultdict(list)

        for position, (word, pos_tag) in enumerate(zip(document.words, [tag for _, tag in document.pos_tags])):
            word_positions[word].append(position)
            self.vocabulary.add(word)

            if position < len(document.lemmas):
                lemma = document.lemmas[position]
                lemma_positions[lemma].append(position)

            pos_positions[pos_tag].append(position)

        for word, positions in word_positions.items():
            if word not in self.index:
                self.index[word] = LexicalPostingList()

            word_pos_tags = [document.pos_tags[pos][1] for pos in positions if pos < len(document.pos_tags)]
            word_lemmas = [document.lemmas[pos] for pos in positions if pos < len(document.lemmas)]

            self.index[word].add_document_with_lexical_info(doc_id, positions, word_pos_tags, word_lemmas)

        for lemma, positions in lemma_positions.items():
            if lemma not in self.lemma_index:
                self.lemma_index[lemma] = LexicalPostingList()

            lemma_pos_tags = [document.pos_tags[pos][1] for pos in positions if pos < len(document.pos_tags)]
            lemma_lemmas = [lemma] * len(positions)

            self.lemma_index[lemma].add_document_with_lexical_info(doc_id, positions, lemma_pos_tags, lemma_lemmas)

        for pos_tag, positions in pos_positions.items():
            if pos_tag not in self.pos_index:
                self.pos_index[pos_tag] = LexicalPostingList()

            # Get the word form information at the position corresponding to this part of speech
            pos_lemmas = [document.lemmas[pos] for pos in positions if pos < len(document.lemmas)]

            self.pos_index[pos_tag].add_document_with_lexical_info(doc_id, positions, [pos_tag] * len(positions), pos_lemmas)

    def search_by_lemma(self, lemma: str) -> Set[int]:
        """Search by lemmatization"""
        lemma = lemma.lower()
        if lemma in self.lemma_index:
            return self.lemma_index[lemma].get_documents()
        return set()

    def search_by_pos(self, pos_tag: str) -> Set[int]:
        """Search by part of speech"""
        if pos_tag in self.pos_index:
            return self.pos_index[pos_tag].get_documents()
        return set()

    def search_lemma_multiple_words(self, lemmas: List[str], operation: str = "and") -> Set[int]:
        """Multi-word lemmatization search"""
        if not lemmas:
            return set()

        if operation == "and":
            result = self.search_by_lemma(lemmas[0])
            for lemma in lemmas[1:]:
                result = result.intersection(self.search_by_lemma(lemma))
                if not result:
                    break
        else:
            result = set()
            for lemma in lemmas:
                result = result.union(self.search_by_lemma(lemma))

        return result

# =============================================================================
# Module 5: Word Vector Reverse Index
# =============================================================================

class SimpleWordVectorizer:
    """Simplified word vector implementation"""
    def __init__(self, vector_size: int = 100):
        self.vector_size = vector_size
        self.word_vectors = {}
        self.vocabulary = set()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        self.is_trained = False

    def train(self, sentences: List[List[str]]):
        """Train simplified word vectors"""
        print("Use a simplified method to train word vectors...")

        for sentence in sentences:
            self.vocabulary.update(sentence)

        corpus = [' '.join(sentence) for sentence in sentences]
        # Train the TF-IDF vectorizer (for feature extraction)
        self.tfidf_vectorizer.fit_transform(corpus)

        feature_names = self.tfidf_vectorizer.get_feature_names_out()

        for word in self.vocabulary:
            if word in feature_names:
                word_idx = list(feature_names).index(word)
                vector = np.random.normal(0, 0.1, self.vector_size)
                vector[word_idx % self.vector_size] += 1.0
                self.word_vectors[word] = vector / np.linalg.norm(vector)
            else:
                vector = np.random.normal(0, 0.1, self.vector_size)
                self.word_vectors[word] = vector / np.linalg.norm(vector)

        self.is_trained = True
        print(f"Training is completed, vocabulary size: {len(self.vocabulary)}")

    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """Obtain word vectors"""
        return self.word_vectors.get(word.lower())

    def similarity(self, word1: str, word2: str) -> float:
        """Calculate word similarity"""
        vec1 = self.get_vector(word1)
        vec2 = self.get_vector(word2)

        if vec1 is not None and vec2 is not None:
            return float(cosine_similarity([vec1], [vec2])[0][0])
        return 0.0

    def most_similar(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """Find the most similar words"""
        word_vec = self.get_vector(word)
        if word_vec is None:
            return []

        similarities = []
        for other_word, other_vec in self.word_vectors.items():
            if other_word != word.lower():
                sim = float(cosine_similarity([word_vec], [other_vec])[0][0])
                similarities.append((other_word, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topn]

class VectorInvertedIndex(LexicalInvertedIndex):
    """Reverse index based on word vectors"""
    def __init__(self, vector_model_type: str = "simple", vector_size: int = 100):
        super().__init__()
        self.vector_model_type = vector_model_type
        self.vector_size = vector_size
        self.word_vectors = None
        self.document_vectors = {}
        self.similarity_threshold = 0.6

        self._initialize_vector_model()

    def _initialize_vector_model(self):
        """Initialize the word vector model"""
        print("Use a simplified word vector model...")
        self.word_vectors = SimpleWordVectorizer(self.vector_size)

    def build_index(self, texts: List[str]):
        """Build vector index"""
        print("Building the word vector reverse index...")

        super().build_index(texts)

        self._train_vector_model(texts)
        self._build_document_vectors()

        print("The construction of the word vector index is completed.！")

    def _train_vector_model(self, texts: List[str]):
        """Train a word vector model"""
        sentences = []
        for i, _ in enumerate(texts):
            doc = self.documents[i]
            sentences.append(doc.words)

        self.word_vectors.train(sentences)

    def _build_document_vectors(self):
        """Build document vectors"""
        print("Build document vectors...")

        for doc_id, document in self.documents.items():
            doc_vector = self._get_average_word_vector(document.words)
            if doc_vector is not None:
                self.document_vectors[doc_id] = doc_vector

    def _get_average_word_vector(self, words: List[str]) -> Optional[np.ndarray]:
        """Get the average word vector as the document vector"""
        vectors = []

        for word in words:
            vector = self.word_vectors.get_vector(word)
            if vector is not None:
                vectors.append(vector)

        if vectors:
            return np.mean(vectors, axis=0)
        return None

    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """Get word vectors"""
        return self.word_vectors.get_vector(word)

    def find_similar_words(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """Find similar words"""
        return self.word_vectors.most_similar(word, topn)

    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Semantic search"""
        query_words = query.lower().split()
        query_vector = self._get_average_word_vector(query_words)

        if query_vector is None:
            return []

        similarities = []
        for doc_id, doc_vector in self.document_vectors.items():
            similarity = cosine_similarity([query_vector], [doc_vector])[0][0]
            similarities.append((doc_id, float(similarity)))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def expanded_search(self, query: str, expansion_count: int = 3) -> Set[int]:
        """Expand search"""
        query_words = query.lower().split()
        expanded_words = set(query_words)

        for word in query_words:
            similar_words = self.find_similar_words(word, expansion_count)
            for similar_word, similarity in similar_words:
                if similarity > self.similarity_threshold:
                    expanded_words.add(similar_word)

        print(f"Original query term: {query_words}")
        print(f"Expanded vocabulary: {expanded_words}")

        return self.search_multiple_words_or(list(expanded_words))

    def hybrid_search(self, query: str, weights: Dict[str, float] = None) -> List[Tuple[int, float]]:
        """Mixed search"""
        if weights is None:
            weights = {"keyword": 0.4, "semantic": 0.6}

        keyword_results = self.ranked_search(query, top_k=10)
        keyword_scores = {doc_id: score for doc_id, score in keyword_results}

        semantic_results = self.semantic_search(query, top_k=10)
        semantic_scores = {doc_id: score for doc_id, score in semantic_results}

        all_docs = set(keyword_scores.keys()) | set(semantic_scores.keys())
        hybrid_scores = []

        for doc_id in all_docs:
            keyword_score = keyword_scores.get(doc_id, 0)
            semantic_score = semantic_scores.get(doc_id, 0)

            if keyword_results:
                max_keyword = max(score for _, score in keyword_results)
                keyword_score = keyword_score / max_keyword if max_keyword > 0 else 0

            if semantic_results:
                max_semantic = max(score for _, score in semantic_results)
                semantic_score = semantic_score / max_semantic if max_semantic > 0 else 0

            hybrid_score = (weights["keyword"] * keyword_score +
                          weights["semantic"] * semantic_score)

            hybrid_scores.append((doc_id, hybrid_score))

        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        return hybrid_scores

# =============================================================================
# Main function and demonstration program
# =============================================================================

def demo_all_modules():
    """
    Demonstrate the functions of all seven modules
    """
    print("=" * 60)
    print("🚀 Complete Demonstration")
    print("=" * 60)

    # =========================================================================
    # Module1: (TextCollector)
    # =========================================================================
    print("\n📰 1. TextCollector")
    print("-" * 30)
    collector = TextCollector()
    # Collect real texts from Project Gutenberg
    texts = collector.collect_texts()


    if not texts:
        print("❌ The text collection failed and the demonstration cannot continue.")
        return

    print(f"✅ Successfully collected {len(texts)} segments of text, ready for subsequent analysis")

    # =========================================================================
    # Module 2: Demonstration of Fixed Phrase Finder
    # =========================================================================
    print("\n🔍 2. Fixed collocation search module")
    print("-" * 30)
    finder = CollocationFinder()
    finder.analyze_texts(texts)

    # Demonstrate hash table lookup (the fastest way to look up)
    hash_collocations = finder.find_collocations_hash(min_frequency=2)
    print(f"📊 High - frequency fixed collocations found using a hash table (the first 5):")
    for phrase, freq in hash_collocations[:5]:
        print(f"  📌 '{phrase}': {freq} ")

    # Demonstrate prefix tree pattern matching
    patterns = ["sewing machine", "of the"]
    trie_results = finder.find_collocations_trie(patterns)
    if trie_results:
        print(f"🌳 Prefix tree pattern matching result:")
        for phrase, freq in trie_results:
            print(f"  🎯 '{phrase}': {freq} ")

    # =========================================================================
    # Module 3: Basic Inverted Index Demonstration
    # =========================================================================
    print("\n📚 3. Basic Reverse Index Module")
    print("-" * 30)
    basic_index = BasicInvertedIndex()
    basic_index.build_index(texts)

    # Demonstrate word search
    docs = basic_index.search_single_word("Shirt")
    print(f"🔍 Word Search 'Shirt': Find the document {docs}")

    # Demonstrate phrase search (position verification)
    docs = basic_index.phrase_search("he had")
    print(f"📝 Phrase search 'he had': Find the document {docs}")

    #Multi - word AND Search Demonstration
    and_query = ["had", "awakened"]
    and_results = basic_index.search_multiple_words_and(and_query)
    print(f"\nMulti - word AND search: {and_query}")
    print(f"Result document ID: {and_results}")

    #Multi - word OR Search Demonstration
    or_query = ["great", "sisterhood"]
    or_results = basic_index.search_multiple_words_or(or_query)
    print(f"\nMulti - word OR search: {or_query}")
    print(f"Result document ID: {or_results}")

    # Demonstrate TF-IDF ranking search
    ranked_results = basic_index.ranked_search("they were", top_k=3)
    if ranked_results:
        print(f"📈 TF-IDF ranking search 'they were':")
        for doc_id, score in ranked_results:
            print(f"  📄 document {doc_id}: TF-IDF score: {score:.4f}")

    # =========================================================================
    # Module 4: Demonstration of Lexical Analysis Inverted Index
    # =========================================================================
    print("\n🔤 4. Lexical Analysis Reverse Index Module")
    print("-" * 30)
    lexical_index = LexicalInvertedIndex()
    lexical_index.build_index(texts)

    # Demonstrate lemmatization search
    docs = lexical_index.search_by_lemma("Shirt")
    print(f"🔄 Lemmatization search 'Shirt': Find the document {docs}")

    # Demonstrate part-of-speech search
    noun_docs = lexical_index.search_by_pos("NN")
    print(f"📝 Part of speech search（NN）: Found {len(noun_docs)} documents")

    # =========================================================================
    # Module 5: Demonstration of Word Vector Inverse Index
    # =========================================================================
    print("\n🧠 5. Word vector reverse index module")
    print("-" * 30)
    # Use a smaller vector dimension to speed up the demonstration.
    vector_index = VectorInvertedIndex(vector_model_type="simple", vector_size=50)
    vector_index.build_index(texts)

    # Demonstrate semantic search
    results = vector_index.semantic_search("Sewing Machine", top_k=3)
    print("🎯 Semantic search 'Sewing Machine':")
    for doc_id, score in results:
        print(f"  🧠 document {doc_id}: Semantic similarity {score:.4f}")

    # Demonstrate the search for similar words
    similar_words = vector_index.find_similar_words("workmen", topn=3)
    if similar_words:
        print(f"🔗 Words similar to 'workmen':")
        for word, similarity in similar_words:
            print(f"  💡 '{word}': Similarity {similarity:.4f}")

    # =========================================================================
    # Demonstration summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("🎉 All module demonstrations are completed！")

if __name__ == "__main__":

    demo_all_modules()
