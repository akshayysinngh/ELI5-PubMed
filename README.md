# ELI5-Pubmed

ELI5 is short for “Explain Like I'm 5,” a request for a simple explanation to a complicated question or problem.

## Why Do We Need This? 
The project aims to make the insights from medical research papers accessible to all by simplifying complex jargon and dense scientific language. Research papers, often written by and for medical professionals, can be challenging for the general public to understand due to their technical nature. This project extracts abstracts from research papers, translates them into easy-to-understand language, and visualizes any embedded data or conclusions. By doing so, it democratizes access to scientific knowledge, allowing a broader audience to grasp and benefit from medical research findings without needing a biostatistics background. 

## How Does This Work?
This process initiates by extracting text from a research paper's abstract section from the URL using BeautifulSoup, segmenting the text into manageable portions. Next, it employs OpenAI's GPT 3.5 model, using Langchain, to convert the text into straightforward, jargon-free explanations. Concurrently, it identifies data within the abstract, structuring it into a clear, tabulated format. The final step involves employing another prompt to generate a visual representation of this data to allow users to quickly comprehend the research findings at a glance.


### Example 
#### Effect of Weekly Subcutaneous Semaglutide (Ozempic) vs Daily Liraglutide (Victoza and Saxenda) on Body Weight in Adults With Overweight or Obesity Without Diabetes: The STEP 8 Randomized Clinical Trial 
URL - https://pubmed.ncbi.nlm.nih.gov/35015037/

#### Output: 
<img width="970" alt="Summary_ _data" src="https://github.com/akshayysinngh/ELI5-Pubmed/assets/91548001/bb6e52ec-0f2f-4669-aaa5-caaaea3cc16d">
<img width="971" alt="data_v" src="https://github.com/akshayysinngh/ELI5-Pubmed/assets/91548001/891f4e8b-c99c-4d2f-b127-f54918aa1da2">
