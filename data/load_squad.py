from datasets import load_dataset
import os
import csv

# Load SQuAD dataset
dataset = load_dataset("squad")

# Set paths
folder_for_documents = 'data/documents'
folder_for_queries = 'data/queries'

os.makedirs(folder_for_documents, exist_ok=True)
os.makedirs(folder_for_queries, exist_ok=True)

# Extract 20 unique paragraphs from the Notre Dame article
seen_contexts = set()
unique_items = []

for item in dataset['train']:
    if 'Notre_Dame' in item['title'] or 'Notre Dame' in item['title']:
        ctx = item['context']
        if ctx not in seen_contexts:
            seen_contexts.add(ctx)
            unique_items.append(item)
    if len(unique_items) == 20:
        break

# Save 20 unique documents
for i, item in enumerate(unique_items):
    title = item['title']
    text = item['context']
    filename = f'{folder_for_documents}/doc_{i:02d}.txt'
    with open(filename, 'w') as f:
        f.write(f"Title: {title}\n\n{text}")

print(f"Created {len(unique_items)} unique documents")

# Save 30 simple queries from the same unique items
# Each unique paragraph has multiple questions — collect one per paragraph
csv_filename = f'{folder_for_queries}/test_queries.csv'
with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['query_id', 'query', 'complexity', 'ground_truth'])

    query_id = 0
    for item in unique_items:
        ctx = item['context']
        # Get all questions for this paragraph
        for entry in dataset['train']:
            if entry['context'] == ctx and query_id < 30:
                writer.writerow([
                    query_id,
                    entry['question'],
                    'simple',
                    entry['answers']['text'][0]
                ])
                query_id += 1
                break  # one question per paragraph to keep variety

print(f"Created {query_id} simple queries")
print("Done! Documents and queries saved.")
