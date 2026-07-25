# Interview Notes — Production Tokenizer Design Patterns

## 1. Registry vs If-Else Chains
**Question**: Why use a Registry pattern for tokenizers instead of simple if-else branches?
- **Open-Closed Principle (OCP)**: A Registry allows developers to extend the supported algorithms (e.g. adding WordPiece or a custom regex normalizer) at runtime without modifying the factory or the service layers.
- **Maintainability**: Decouples the mapping logic from instantiation logic, preventing a central factory file from growing indefinitely.

## 2. Decoupling with a Service Layer
**Question**: What is the purpose of the TokenizerService wrapper?
- It shields downstream modules (such as embedding networks or generation APIs) from the underlying tokenizer implementation details.
- Standardizes parameter formats (like `add_special_tokens` and padding lengths) so that swapping BPE for WordPiece or SentencePiece is a zero-code-change operation for the Transformer model.

## 3. High-Performance Tokenizer Profiling
**Question**: How do you analyze tokenizer bottlenecks?
- **Execution Time**: Measure throughput in terms of processed tokens per second.
- **Memory Overhead**: Use Python's `tracemalloc` to track peak heap allocations. Standardize string buffer copies to minimize memory usage on large text inputs.
