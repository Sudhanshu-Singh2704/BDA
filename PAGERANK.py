from collections import defaultdict


DAMPING_FACTOR = 0.85
MAX_ITER = 10
TOLERANCE = 1e-6



def pagerank_map(url, pagerank, outlinks):
    mapped = []
    if outlinks:
        share = pagerank / len(outlinks)
        for outlink in outlinks:
            mapped.append((outlink, share))
    mapped.append((url, outlinks))
    return mapped



def pagerank_reduce(url, values, N):
    outlink_list = []
    total_pr = 0.0

    for v in values:
        if isinstance(v, list):
            outlink_list = v
        else:
            total_pr += v

    new_pr = ((1 - DAMPING_FACTOR) / N) + (DAMPING_FACTOR * total_pr)
    return (url, new_pr, outlink_list)


def pagerank_mapreduce(graph, max_iters=MAX_ITER):
    N = len(graph)
    ranks = {url: 1.0 / N for url in graph}

    for _ in range(max_iters):
        mapped_data = defaultdict(list)

        # Map phase
        for url, outlinks in graph.items():
            for key, value in pagerank_map(url, ranks[url], outlinks):
                mapped_data[key].append(value)

        # Reduce phase
        new_ranks = {}
        new_graph = {}
        for url, values in mapped_data.items():
            url, new_pr, outlink_list = pagerank_reduce(url, values, N)
            new_ranks[url] = new_pr
            new_graph[url] = outlink_list

        diff = sum(abs(new_ranks[url] - ranks[url]) for url in ranks)
        ranks = new_ranks
        graph = new_graph
        if diff < TOLERANCE:
            break

    sorted_pages = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
    return sorted_pages


if __name__ == "__main__":
    n = int(input("Enter number of pages: "))
    graph = {}

    print("\nEnter each page and its outlinks (space-separated).")
    print("Example: For page A linking to B and C, type: A B C")
    print("If a page has no outlinks, just type the page name (e.g., D)\n")

    for _ in range(n):
        parts = input("→ ").split()
        page = parts[0]
        outlinks = parts[1:] if len(parts) > 1 else []
        graph[page] = outlinks

    result = pagerank_mapreduce(graph)

    print("\nFinal PageRank Results (sorted):")
    for url, pr in result:
        print(f"{url}: {pr:.6f}")
