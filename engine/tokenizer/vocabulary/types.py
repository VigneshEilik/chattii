from typing import Dict, Tuple

# Type Aliases for code clarity
Token = str
TokenId = int

# Statistics aliases
TokenFrequency = int
MergePair = Tuple[str, str]

# Serialization structures
VocabMap = Dict[Token, TokenId]
SpecialTokensMap = Dict[str, Token]
