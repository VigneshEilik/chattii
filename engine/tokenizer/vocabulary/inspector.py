from typing import Any, Dict


class VocabularyInspector:
    """
    Renders structured vocabulary audit tables and histograms to stdout.
    """

    @staticmethod
    def inspect(loaded_data: Dict[str, Any]) -> None:
        """
        Prints a formatted report of vocabulary specifications and metrics.
        """
        metadata = loaded_data.get("metadata", {})
        vocab = loaded_data.get("vocab", {})
        statistics = loaded_data.get("statistics", {})
        special_tokens = loaded_data.get("special_tokens", {})
        manifest = loaded_data.get("manifest", {})

        print("=" * 60)
        print("                 CHATII VOCABULARY AUDIT REPORT")
        print("=" * 60)
        print(f"Vocabulary Name:   {manifest.get('name', 'default')}")
        print(f"Algorithm:         {metadata.get('algorithm', 'unknown').upper()}")
        print(f"Version:           {metadata.get('version', 'N/A')}")
        print(f"Lifecycle State:   {manifest.get('status', 'N/A').upper()}")
        print(f"Created At:        {metadata.get('created_at', 'N/A')}")
        print("-" * 60)
        print(f"Total Vocab Size:  {statistics.get('vocab_size', len(vocab))} tokens")
        print(f"Reserved Count:    {statistics.get('reserved_tokens_count', 0)} tokens")
        print(f"Unique Count:      {statistics.get('unique_tokens_count', 0)} tokens")
        print(f"Normalization:     {metadata.get('normalization', 'N/A')}")
        print(f"Lowercase Enabled: {metadata.get('lowercase', False)}")
        print(f"Min Freq Threshold: {metadata.get('min_frequency', 1)}")
        print("-" * 60)
        print(f"Average Token Len: {statistics.get('average_token_length', 0.0)} chars")
        print(
            f"Longest Token:     "
            f"'{statistics.get('longest_token', '')}' "
            f"({statistics.get('longest_token_length', 0)} chars)"
        )
        print(
            f"Shortest Token:    "
            f"'{statistics.get('shortest_token', '')}' "
            f"({statistics.get('shortest_token_length', 0)} chars)"
        )
        print(f"Token Entropy:     {statistics.get('token_entropy', 0.0)}")
        print("-" * 60)

        print("Special Tokens Mapping:")
        for name, token in special_tokens.items():
            print(f"  - {name:8} -> {token}")

        print("-" * 60)

        # Sort vocabulary mapping by token ID
        sorted_by_id = sorted(vocab.items(), key=lambda item: item[1])

        print("First 20 Vocabulary Tokens (by ID):")
        for tok, tok_id in sorted_by_id[:20]:
            # Replace whitespace tokens with descriptive tags for visibility
            repr_tok = tok.replace(" ", " [space]")
            print(f"  ID [{tok_id:5}] -> '{repr_tok}'")

        print("=" * 60)
