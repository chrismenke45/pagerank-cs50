import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    linked_pages = corpus[page]
    corpus_length = len(corpus)
    linked_pages_length = len(linked_pages)
    
    if linked_pages_length:
        random_page_probability = (1 / corpus_length) * (1 - damping_factor)
        linked_page_probability = (1 / linked_pages_length) * damping_factor
    else:
        random_page_probability = (1 / corpus_length)
        linked_page_probability = 0
    
    output = {}
    for key in corpus:
        output[key] = random_page_probability
        if key in linked_pages:
            output[key] += linked_page_probability
    return output


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    current_page = random.choice(list(corpus.keys()))
    pages_probabilities = {}
    pages_count = {}
    for i in range(n):
        prob_count = 0
        rgn = random.random()

        if (current_page in pages_count):
            pages_count[current_page] += 1
        else:
            pages_count[current_page] = 1

        if (current_page in pages_probabilities):
            current_page_probabilities = pages_probabilities[current_page]
        else:
            current_page_probabilities = transition_model(corpus, current_page, damping_factor)
            pages_probabilities[current_page] = current_page_probabilities

        for key in current_page_probabilities:
            prob_count += current_page_probabilities[key]
            if (rgn < prob_count):
                current_page = key
                break
    
    output_ranks = {}
    for key in pages_count:
        output_ranks[key] = pages_count[key] / n

    return output_ranks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page_ranks = {}
    page_count = len(corpus)
    starting_rank = 1 / page_count
    pages_that_mention_pages = {}
    for page in corpus:
        page_ranks[page] = starting_rank
        for referenced_page in corpus[page]:
            if referenced_page in pages_that_mention_pages:
                pages_that_mention_pages[referenced_page].add(page)
            else:
                pages_that_mention_pages[referenced_page] = {page}

    pages_list = list(corpus.keys())
    current_count = 0
    convergence_count = 0
    while True:
        if convergence_count >= page_count: break
        current_page = pages_list[current_count % page_count]
        references_sum = 0
        if current_page in pages_that_mention_pages:
            for reference_page in pages_that_mention_pages[current_page]:
                references_sum += damping_factor * (page_ranks[reference_page] / (len(corpus[reference_page]) or page_count))
        else:
            references_sum += damping_factor * (page_ranks[reference_page] / page_count)
        new_rank = ((1 - damping_factor) / page_count) + references_sum
        
        if abs(page_ranks[current_page] - new_rank) <= 0.001:
            convergence_count += 1
        else:
            convergence_count = 0
        page_ranks[current_page] = new_rank
        current_count += 1

    return page_ranks


if __name__ == "__main__":
    main()
