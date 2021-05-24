# iter-opt-filter &emsp; [![Latest Version]][https://crates.io/crates/iter-opt-filter] [![Docs.rs]][https://docs.rs/iter-opt-filter]

This crate adds an optional filter to iterators. The problem this attempts to solve is the combination of multiple filters that can be enabled/disabled at runtime.

## Example

```rust
use iter_opt_filter::IteratorOptionalFilterExt;

let mut iter = (0..3)
    .optional_filter(Some(|&item: &usize| item % 2 == 0))
    .optional_filter(None::<fn(&usize) -> bool>)
    .optional_filter(Some(|&item: &usize| item > 1));

assert_eq!(iter.next(), Some(2));
assert_eq!(iter.next(), None);
```