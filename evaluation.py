# Create test pairs: known similar and dissimilar inputs
test_cases = [
    # Similar pairs (should score > 0.8)
    ("need AI for my SaaS product", "looking for AI integration for software"),
    ("urgent budget approved", "ready to buy immediately"),
    
    # Dissimilar pairs (should score < 0.3)
    ("need AI for my SaaS", "no interest just browsing"),
    ("urgent purchase needed", "not sure maybe later"),
]

def evaluate_similarity(test_cases):
    for text1, text2 in test_cases:
        emb1 = generate_embedding(text1)
        emb2 = generate_embedding(text2)
        score = np.dot(emb1, emb2)  # cosine since normalized
        print(f"Score: {score:.3f} | {text1[:30]}... vs {text2[:30]}...")
