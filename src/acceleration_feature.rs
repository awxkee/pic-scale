#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub(crate) enum AccelerationFeature {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    Neon,
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    Sse,
    Native
}