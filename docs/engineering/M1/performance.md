# Performance Optimization

## 1. Frequency Registry Caching
Instead of scanning the raw corpus string repeatedly, the corpus is pre-aggregated into a dictionary of word sequence types to frequencies (`Dict[Tuple[str, ...], int]`). The pair counts are computed in $O(U)$ rather than $O(N)$ where $U \ll N$.

## 2. Fast Decoding
Decoders store a reverse mapping array or dictionary `self._id_to_token` enabling $O(1)$ token string resolution.

## 3. SHA-256 Checksum Validation
To avoid runtime errors or corrupted model weights, model loading uses SHA-256 signatures to verify file integrity.
