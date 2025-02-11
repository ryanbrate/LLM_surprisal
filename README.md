Jupyter notebooks are provided to reproduce the analysis: assembling the texts, extracting quotations, getting LLM logits for the quotations, and forming the analysis on LLM surprisals.  Additionally, the resultant quotation collections and corresponding LLM normalised logits are also made available.

# Extracting quotation sets

The notebooks referenced in the following sub-sections  use python version 3.11.4 and the following non-standard packages:

- pandas=2.23

- numpy=1.25.2

- spacy=3.8.2

- nltk=3.9.1

- torch=2.5.1

- transformers=4.46.1

(note that 'torch' and 'transformers' are merely used to apply the gpt2 tokenizer, such to identify quotation spans over 1024 tokens. Hence, the versioning of dependent packages is not critical.)

# Building the American literature fiction corpus

## Identify and extract American literature fiction texts from 

The American literature corpus is assembled from Project Gutenberg: all texts with Library of Congress designation 'PS' and language 'en', which also contain the word $n$-word. This collection of book urls can be found [here](PG/PG_sample/book_urls.csv), based on the [Project Gutenberg catalogue](https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv). The catalogue snapshot at the time of analysis is downloaded to [here](PG/PG_sample/pg_catalog.csv)

[main.ipynb](PG/PG_sample/main.ipynb) downloads the PG catalogue, identifies the en + PS subset of texts, builds a list of book urls, downloads the books to a specified location.

## Extract $n$-word and normative reference attributed quotations from the American literature fiction texts

See [extract_quotations_via_spaCy.ipynb](PG/extract_quotes_via_spaCy/extract_quotations_via_spaCy.ipynb), which identifies all corresponding file id, quote, manner, speaker instances from the American literature fiction texts. 


Note: we assume latin-1 encoding for the book collections.

Note: we limit quotations to those with no more than 1024 characters


Extracted (file id, quote, manner, speaker) instances are saved as quotes\_5Jul.json and the blacklisted quote indices where quotations exceed 1024 characters are saved as quotes\_blacklist.json

Both of the resultant files can be found here: 
- [quotes\_5Jul.json](https://drive.google.com/file/d/1DHAV1krlTfV1ROGVAV-Z34epj30SNB7c/view?usp=sharing)
- [quotes\_blacklist.json](https://drive.google.com/file/d/1XANN0xWDKXrgBhhPs3UNTJOZKUwHFh4W/view?usp=sharing)


# Building the Library of Congress (LOC) newspaper corpus of quotations

See [notebook](LOC/news_quotatations_extraction.ipynb) for the identification of LOC newspaper pages referencing $n$-word attributed quotations, and their automatic extraction. The extracted quotations have then been manually corrected.

The manually corrected LOC extracted quotations are found [here](LOC/tuples_news.json) 

# Getting LLM suprisal scores for the quotation successive sub-word tokens

This has been done on the snellius server using Llama 3.1.

Python 3.11.4 with the following module requirements: 

## LOC newspaper collection

The Llama 3.1 softmaxed logits wrt., the $n$-word attributed quotations from the Library of Congress American news collection, is available [here](LOC/Snellius/LOC_llama3.1_70B/chains_llama3.1_70B.json), produced via [this script](LOC/Snellius/LOC_llama3.1_70B/score_with_llama3.py)

# Analysis

## RQ1

RQ: In American literature fiction, what targeted non-canonical linguistic variations contribute most to the observed deviation in LLM surprisal, between speakers referenced by '$n$-word' and normative reference speakers?

This deviation is measured as $S_{n-word } - S_{normative\ reference\ speaker}$, 
where S is the mean LLM surprisal of words in context wrt., an attributed speaker referent.

LLM surprisal is estimated according to normalised logits. Normalised logits are estimated according to LLama3.1 70B according to [this script](PG/Snellius/mwcgln/llama3.1_70B/score_with_llama3.py), applied to the extracted American literature quotations corresponding to $n$-word and the normative reference speakers, minus the blacklisted quotations exceeding 1024 sub-tokens. The resultant normalised logits are found [here](PG/Snellius/mwcgln/llama3.1_70B/), as the output of [this script](PG/Snellius/mwcgln/llama3.1_70B/score_with_llama3.py). The script has been run on the snellius server, using 4No. H100 cpus, run on python 3.11.4 with the following pip requirements: [requirements.txt](snellius_requirements.txt)


In [RQ1.ipynb](Analysis/llama3.1_70B/RQ1.ipynb), the following is done:

- determine $S_{n-word} - S_{normative\ reference\ speaker}$, for each normative reference speaker

- determine posterior estimates of $Z_{n-word} - Z_{normative\ reference\ speaker}$, where $Z_{speaker}$ is the posterior estimate of latent population mean word LLM surprisal, on the basis that the observed quotations are representative 

- identify out of dictionary (OOD) words. See [here](Analysis/llama3.1_70B/RQ1_ood_words/) for the OOD words by speaker referent, together with manually identified canonical form the word approximates.

- identify the phonological variations implied by the non-canonical form of the OOD words, as compared to the manually identified canonical form.

- for each targeted variation (each identified phonological variation and targeted dialectical), build a [collection of quotations with each separate variation corrected](Analysis/llama3.1_70B/RQ1_downstream/all_corrected.json), and [corresponding normalised sub-word logits](Analysis/llama3.1_70B/RQ1_downstream/chains_llama3.1_70B_all_corrected.json) via [script](Analysis/llama3.1_70B/RQ1_downstream/score_with_llama3.py) run on snellius, python3 3.11.4, with [these pip requirements](snellius_requirements.txt). For each variation, calculate the contribution of the variation to $S_{n-word} - S_{normative\ reference\ speaker}$, and rank the variations by this contribution.


Note: pymc traces can be found [here](https://drive.google.com/drive/folders/1Kkpda-gFMzFRRZj32ugjYfW-O_6-5NLR?usp=drive_link)

## RQ2

RQ: which named speakers in American literature fiction, are shown to be most strongly associated with those same linguistic variations that distinguish $n$-word attributed quotations from normative reference speakers.

We target the variations identified in RQ1 as explaining the greatest variation in $S_{n-word } - S_{normative\ reference\ speaker}$, as per Table 1 of the research paper. 

In [RQ2.ipynb](Analysis/llama3.1_70B/RQ2.ipynb):

- for each of the targeted variations, we identify quotes in American literature fiction (outside of $n$-word and the normative reference speakers) that share the observed non-canonical word forms which exhibit the targeted variations: [identified quotes](Analysis/llama3.1_70B/RQ2/RQ2_selected_sids.json). 

- we get [normalised logits this set of quotes](Analysis/llama3.1_70B/RQ2/chains_llama3.1_70B_sids.json) via [this script](Analysis/llama3.1_70B/RQ2/score_with_llama3.py) on snellius, python 3.11.4 with [these pip requirements](snellius_requirements.txt)

- we build a [collection of quotations with each separate variation corrected](Analysis/llama3.1_70B/RQ2_downstream/all_corrected.json), and [corresponding normalised sub-word logits](Analysis/llama2.1_70B/RQ2_downstream/chains_llama3.1_70B.json) via [script](Analysis/llama3.1_70B/RQ2_downstream/score_with_llama3.py), run on snellius, python 3.11.4, [these pip requirements](snellius_requirements.txt). For each targeted variation we rank the named speakers according to the variations contribution to mean word LLM surprisal.These rankings are reproduced in Table 4 of the table.

## RQ3 

RQ: According to LLM surprisal, to what extent and due to what targeted linguistic variations, do quotations attributed to speakers referenced by '$n$-word' differ between the American literature fiction and American News domains?

We estimate the [normalised logits](LOC/Snellius/LOC_llama3.1_70B/chains_llama3.1_70B.json) of sub-word tokens for the corresponding LOC newspaper collection extracted quotations, via [this script](LOC/Snellius/LOC_llama3.1_70B/score_with_llama3.py), run on snellius, with python 3.11.4 and with [these pip requirements](snellius_requirements.txt)

In [RQ3.ipynb](Analysis/llama3.1_70B/RQ3.ipynb) we:

- estimate $S_{n-word,\ lit} - S_{n-word,\ news}$, where $S_{n-word,\corpus}$ is the mean LLM surprisal of observed quotations of some corpus of quotations

- estimate $Z_{n-word,\ lit} - Z_{n-word,\ news}$, where $Z_{n-word,\corpus}$ is the mean LLM surprisal of the presumed population of quotations, for which the observed quotations are presumed representative

- identify [out of dictionary words](Analysis/llama3.1_70B/RQ3_ood_words/RQ3_ood_words.json) in the Library of Congress $n$-word attributed quotations, and manually annotate corresponding canonical forms.

- identify phonological variations present in the LOC $n$-word attributed quotations as implied in the textual variations between out-of-dictionary words and the manually proposed canonical forms

- build a [set of quotations corrected each phonological and targeted dialectical variation](Analysis/llama3.1_70B/RQ3_downstream/all_corrected.json). We then get the [corresponding normalised logits](Analysis/llama3.1_70B/RQ3_downstream/chains_llama3.1_70B_all_corrected.json) via [this script](Analysis/llama3.1_70B/RQ3_downstream/score_with_llama3.py), run on snellius python 3.11.4 with [these pip requirements](snellius_requirements.txt)

- for each variation in the American literature fiction quotations attributed to $n$-word speaker references, estimate the difference in contribution to the mean word LLM surprisal of the American literature fiction quotations versus those of the LOC quotations, attributed to $n$-word speaker referents. This is reproduced as Table 5 in the paper.


Note: pymc traces can be found [here](https://drive.google.com/drive/folders/1-CREwprFlCkLvRKwcEQM5LzLPtr5rD9n?usp=drive_link)
