//! This library adds the [optional_filter] which is provided for all [Iterators][Iterator] by the [IteratorOptionalFilterExt] extension trait.
//! 
//! The [optional_filter] takes an `Option<fn(&item) -> bool>`, which allows for easy conditional filtering.
//! This however comes at a performance cost compared to a normal [filter][Iterator::filter] or the inner [Iterator] by itself.
//! But it is generally faster than a `Box<dyn Iterator>`.
//!
//! See examples on the [optional_filter] method.
//!
//! [optional_filter]: IteratorOptionalFilterExt::optional_filter

#![cfg_attr(test, feature(test))]

use std::fmt;
use std::iter::FusedIterator;

/// Extension trait for adding the [optional_filter][IteratorOptionalFilterExt::optional_filter] method to iterators.
pub trait IteratorOptionalFilterExt<P: FnMut(&Self::Item) -> bool>: Iterator + Sized {
    /// Filters the iterator with the predicate, like [filter][Iterator::filter]. If the predicate is `None`, all items will be returned.
    /// 
    /// # Examples
    /// 
    /// Basic usage:
    /// 
    /// ```
    /// use iter_opt_filter::IteratorOptionalFilterExt;
    ///
    /// let mut iter = (0..3).optional_filter(Some(|&item: &usize| item % 2 == 0));
    /// assert_eq!(iter.next(), Some(0));
    /// assert_eq!(iter.next(), Some(2));
    /// assert_eq!(iter.next(), None);
    /// 
    /// let mut iter = (0..3).optional_filter(None::<fn(&usize) -> bool>);
    /// assert_eq!(iter.next(), Some(0));
    /// assert_eq!(iter.next(), Some(1));
    /// assert_eq!(iter.next(), Some(2));
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// Because the type stays the same, regardless of the predicate being `Some` or `None`, the filters are easily chainable:
    /// 
    /// ```
    /// use iter_opt_filter::IteratorOptionalFilterExt;
    /// 
    /// let mut iter = (0..3)
    ///     .optional_filter(Some(|&item: &usize| item % 2 == 0))
    ///     .optional_filter(None::<fn(&usize) -> bool>)
    ///     .optional_filter(Some(|&item: &usize| item > 1));
    /// assert_eq!(iter.next(), Some(2));
    /// assert_eq!(iter.next(), None);
    /// ```
    fn optional_filter(self, predicate: Option<P>) -> OptionalFilter<Self, P> {
        OptionalFilter {
            iter: self,
            predicate,
        }
    }
}

impl<I, P> IteratorOptionalFilterExt<P> for I
where
    I: Iterator,
    P: FnMut(&I::Item) -> bool
{}
/// An iterator that optionally filters the items with a predicate.
/// This `struct` is created by the [optional_filter][IteratorOptionalFilterExt::optional_filter] method provided by the [IteratorOptionalFilterExt] extension trait.
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[derive(Clone)]
pub struct OptionalFilter<I, P> {
    iter: I,
    predicate: Option<P>,
}

impl<I, P> Iterator for OptionalFilter<I, P>
where
    I: Iterator,
    P: FnMut(&I::Item) -> bool
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.predicate {
            Some(predicate) => self.iter.find(predicate),
            None => self.iter.next(),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.iter.size_hint();
        if self.predicate.is_some() {
            (0, upper)
        }
        else {
            (lower, upper)
        }
    }

    // Specialization from std::iter::Filter
    #[inline]
    fn count(self) -> usize {
        #[inline]
        fn to_usize<T>(mut predicate: impl FnMut(&T) -> bool) -> impl FnMut(T) -> usize {
            move |x| predicate(&x) as usize
        }
        
        match self.predicate {
            Some(predicate) => self.iter.map(to_usize(predicate)).sum(),
            None => self.iter.count(),
        }
    }
}

impl<I, P> DoubleEndedIterator for OptionalFilter<I, P>
where
    I: DoubleEndedIterator,
    P: FnMut(&I::Item) -> bool
{
    fn next_back(&mut self) -> Option<Self::Item> {
        match &mut self.predicate {
            Some(predicate) => self.iter.rfind(predicate),
            None => self.iter.next_back(),
        }
    }
}

impl<I, P> FusedIterator for OptionalFilter<I, P>
where
    I: FusedIterator,
    P: FnMut(&I::Item) -> bool
{}

impl<I: fmt::Debug, P> fmt::Debug for OptionalFilter<I, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OptionalFilter")
            .field("iter", &self.iter)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    extern crate test;

    #[test]
    fn const_enabled() {
        let output: Vec<_> = (0..10).optional_filter(Some(|&item: &usize| item % 2 == 0)).collect();
        assert_eq!(output, vec![0, 2, 4, 6, 8])
    }

    #[test]
    fn const_disabled() {
        let output: Vec<_> = (0..10).optional_filter(None::<fn(&usize) -> bool>).collect();
        assert_eq!(output, (0..10).collect::<Vec<_>>())
    }

    #[test]
    fn double_ended_enabled() {
        let mut iter = (0..10).optional_filter(Some(|&item: &usize| item % 2 == 0));
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next_back(), Some(8));
        assert_eq!(iter.next_back(), Some(6));
        assert_eq!(iter.next_back(), Some(4));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn nested_const() {
        let output: Vec<_> = (0..10)
            .optional_filter(Some(|&item: &usize| item % 2 == 0))
            .optional_filter(None::<fn(&usize) -> bool>)
            .optional_filter(Some(|&item: &usize| item % 3 == 0))
            .collect();
        assert_eq!(output, vec![0, 6])
    }

    // bool_to_option is still unstable
    fn bool_to_opt<T>(b: bool, some: T) -> Option<T> {
        if b {
            Some(some)
        }
        else {
            None
        }
    }
    
    fn filter_nested(cond1: bool, cond2: bool) -> Vec<usize> {
        let cond1 = test::black_box(cond1);
        let cond2 = test::black_box(cond2);
        (0..10)
            .optional_filter(bool_to_opt(cond1, |&item: &usize| item % 2 == 0))
            .optional_filter(bool_to_opt(cond2, |&item: &usize| item % 3 == 0))
            .collect()
    }

    #[test]
    fn nested_false_false() {
        assert_eq!(filter_nested(false, false), (0..10).collect::<Vec<usize>>());
    }

    #[test]
    fn nested_false_true() {
        assert_eq!(filter_nested(false, true), vec![0, 3, 6, 9]);
    }

    #[test]
    fn nested_true_false() {
        assert_eq!(filter_nested(true, false), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn nested_true_true() {
        assert_eq!(filter_nested(true, true), vec![0, 6]);
    }

    mod benches {
        use super::*;
        use test::Bencher;

        trait IteratorOptionalFilterBoxedExt<'a, P: FnMut(&Self::Item) -> bool>
        where
            Self: Iterator + Sized + 'a,
            P: 'a
        {
            fn optional_filter_boxed(self, predicate: Option<P>) -> Box<dyn Iterator<Item = Self::Item> + 'a>
            {
                match predicate {
                    Some(predicate) => Box::new(self.filter(predicate)),
                    None => Box::new(self),
                }
            }
        }

        impl<'a, I, P> IteratorOptionalFilterBoxedExt<'a, P> for I
        where
            I: Iterator + 'a,
            P: FnMut(&I::Item) -> bool + 'a
        {}
        
        mod collect_vec {
            use super::*;
            #[bench]
            fn disabled_baseline(b: &mut Bencher) {
                b.iter(|| {
                    let n: usize = test::black_box(1_000_000);
                    (0..n).collect::<Vec<_>>()
                })
            }
        
            #[bench]
            fn disabled_blackbox(b: &mut Bencher) {
                b.iter(|| {
                    let n: usize = test::black_box(1_000_000);
                    let f: Option<fn(&usize) -> bool> = test::black_box(None);
                    (0..n).optional_filter(f).collect::<Vec<_>>()
                })
            }
        
            #[bench]
            fn disabled_blackbox_boxed(b: &mut Bencher) {
                b.iter(|| {
                    let n: usize = test::black_box(1_000_000);
                    let f: Option<fn(&usize) -> bool> = test::black_box(None);
                    (0..n).optional_filter_boxed(f).collect::<Vec<_>>()
                })
            }
        
            #[bench]
            fn disabled_const(b: &mut Bencher) {
                b.iter(|| {
                    let n: usize = test::black_box(1_000_000);
                    (0..n).optional_filter(None::<fn(&usize) -> bool>).collect::<Vec<_>>()
                })
            }
        
            #[bench]
            fn disabled_const_boxed(b: &mut Bencher) {
                b.iter(|| {
                    let n: usize = test::black_box(1_000_000);
                    (0..n).optional_filter_boxed(None::<fn(&usize) -> bool>).collect::<Vec<_>>()
                })
            }
        
            #[bench]
            fn enabled_baseline(b: &mut Bencher) {
                b.iter(|| {
                    let n: usize = test::black_box(1_000_000);
                    (0..n).filter(|&item| item % 2 == 0).collect::<Vec<_>>()
                })
            }
        
            #[bench]
            fn enabled_blackbox(b: &mut Bencher) {
                b.iter(|| {
                    let n: usize = test::black_box(1_000_000);
                    let f: Option<fn(&usize) -> bool> = test::black_box(Some(|&item| item % 2 == 0));
                    (0..n).optional_filter(f).collect::<Vec<_>>()
                })
            }
        
            #[bench]
            fn enabled_blackbox_boxed(b: &mut Bencher) {
                b.iter(|| {
                    let n: usize = test::black_box(1_000_000);
                    let f: Option<fn(&usize) -> bool> = test::black_box(Some(|&item| item % 2 == 0));
                    (0..n).optional_filter_boxed(f).collect::<Vec<_>>()
                })
            }
        
            #[bench]
            fn enabled_const(b: &mut Bencher) {
                b.iter(|| {
                    let n: usize = test::black_box(1_000_000);
                    let f: Option<fn(&usize) -> bool> = Some(|&item| item % 2 == 0);
                    (0..n).optional_filter(f).collect::<Vec<_>>()
                })
            }
        
            #[bench]
            fn enabled_const_boxed(b: &mut Bencher) {
                b.iter(|| {
                    let n: usize = test::black_box(1_000_000);
                    let f: Option<fn(&usize) -> bool> = Some(|&item| item % 2 == 0);
                    (0..n).optional_filter_boxed(f).collect::<Vec<_>>()
                })
            }
        }

        mod count {
            use super::*;
            #[bench]
            fn disabled_baseline(b: &mut Bencher) {
                b.iter(|| {
                    let n: usize = test::black_box(1_000_000);
                    (0..n).count()
                })
            }
        
            #[bench]
            fn disabled_blackbox(b: &mut Bencher) {
                b.iter(|| {
                    let n: usize = test::black_box(1_000_000);
                    let f: Option<fn(&usize) -> bool> = test::black_box(None);
                    (0..n).optional_filter(f).count()
                })
            }
        
            #[bench]
            fn disabled_blackbox_boxed(b: &mut Bencher) {
                b.iter(|| {
                    let n: usize = test::black_box(1_000_000);
                    let f: Option<fn(&usize) -> bool> = test::black_box(None);
                    (0..n).optional_filter_boxed(f).count()
                })
            }
        
            #[bench]
            fn disabled_const(b: &mut Bencher) {
                b.iter(|| {
                    let n: usize = test::black_box(1_000_000);
                    (0..n).optional_filter(None::<fn(&usize) -> bool>).count()
                })
            }
        
            #[bench]
            fn disabled_const_boxed(b: &mut Bencher) {
                b.iter(|| {
                    let n: usize = test::black_box(1_000_000);
                    (0..n).optional_filter_boxed(None::<fn(&usize) -> bool>).count()
                })
            }
        
            #[bench]
            fn enabled_baseline(b: &mut Bencher) {
                b.iter(|| {
                    let n: usize = test::black_box(1_000_000);
                    (0..n).filter(|&item| item % 2 == 0).count()
                })
            }
        
            #[bench]
            fn enabled_blackbox(b: &mut Bencher) {
                b.iter(|| {
                    let n: usize = test::black_box(1_000_000);
                    let f: Option<fn(&usize) -> bool> = test::black_box(Some(|&item| item % 2 == 0));
                    (0..n).optional_filter(f).count()
                })
            }
        
            #[bench]
            fn enabled_blackbox_boxed(b: &mut Bencher) {
                b.iter(|| {
                    let n: usize = test::black_box(1_000_000);
                    let f: Option<fn(&usize) -> bool> = test::black_box(Some(|&item| item % 2 == 0));
                    (0..n).optional_filter_boxed(f).count()
                })
            }
        
            #[bench]
            fn enabled_const(b: &mut Bencher) {
                b.iter(|| {
                    let n: usize = test::black_box(1_000_000);
                    let f: Option<fn(&usize) -> bool> = Some(|&item| item % 2 == 0);
                    (0..n).optional_filter(f).count()
                })
            }
        
            #[bench]
            fn enabled_const_boxed(b: &mut Bencher) {
                b.iter(|| {
                    let n: usize = test::black_box(1_000_000);
                    let f: Option<fn(&usize) -> bool> = Some(|&item| item % 2 == 0);
                    (0..n).optional_filter_boxed(f).count()
                })
            }
        }

        mod sum {
            use super::*;
            #[bench]
            fn disabled_baseline(b: &mut Bencher) {
                b.iter(|| {
                    let n: usize = test::black_box(1_000_000);
                    (0..n).sum::<usize>()
                })
            }
        
            #[bench]
            fn disabled_blackbox(b: &mut Bencher) {
                b.iter(|| {
                    let n: usize = test::black_box(1_000_000);
                    let f: Option<fn(&usize) -> bool> = test::black_box(None);
                    (0..n).optional_filter(f).sum::<usize>()
                })
            }
        
            #[bench]
            fn disabled_blackbox_boxed(b: &mut Bencher) {
                b.iter(|| {
                    let n: usize = test::black_box(1_000_000);
                    let f: Option<fn(&usize) -> bool> = test::black_box(None);
                    (0..n).optional_filter_boxed(f).sum::<usize>()
                })
            }
        
            #[bench]
            fn disabled_const(b: &mut Bencher) {
                b.iter(|| {
                    let n: usize = test::black_box(1_000_000);
                    (0..n).optional_filter(None::<fn(&usize) -> bool>).sum::<usize>()
                })
            }
        
            #[bench]
            fn disabled_const_boxed(b: &mut Bencher) {
                b.iter(|| {
                    let n: usize = test::black_box(1_000_000);
                    (0..n).optional_filter_boxed(None::<fn(&usize) -> bool>).sum::<usize>()
                })
            }
        
            #[bench]
            fn enabled_baseline(b: &mut Bencher) {
                b.iter(|| {
                    let n: usize = test::black_box(1_000_000);
                    (0..n).filter(|&item| item % 2 == 0).sum::<usize>()
                })
            }
        
            #[bench]
            fn enabled_blackbox(b: &mut Bencher) {
                b.iter(|| {
                    let n: usize = test::black_box(1_000_000);
                    let f: Option<fn(&usize) -> bool> = test::black_box(Some(|&item| item % 2 == 0));
                    (0..n).optional_filter(f).sum::<usize>()
                })
            }
        
            #[bench]
            fn enabled_blackbox_boxed(b: &mut Bencher) {
                b.iter(|| {
                    let n: usize = test::black_box(1_000_000);
                    let f: Option<fn(&usize) -> bool> = test::black_box(Some(|&item| item % 2 == 0));
                    (0..n).optional_filter_boxed(f).sum::<usize>()
                })
            }
        
            #[bench]
            fn enabled_const(b: &mut Bencher) {
                b.iter(|| {
                    let n: usize = test::black_box(1_000_000);
                    let f: Option<fn(&usize) -> bool> = Some(|&item| item % 2 == 0);
                    (0..n).optional_filter(f).sum::<usize>()
                })
            }
        
            #[bench]
            fn enabled_const_boxed(b: &mut Bencher) {
                b.iter(|| {
                    let n: usize = test::black_box(1_000_000);
                    let f: Option<fn(&usize) -> bool> = Some(|&item| item % 2 == 0);
                    (0..n).optional_filter_boxed(f).sum::<usize>()
                })
            }
        }
    }
}
