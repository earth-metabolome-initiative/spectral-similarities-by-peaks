//! Decomposition of `SimilarityConfig::name()` slugs into structured keys
//! so the UI can present each dimension (family, m/z exponent, intensity
//! exponent, entropy weighting) as its own pill row instead of one big
//! dropdown.

#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use std::collections::BTreeSet;

/// Top-level similarity family extracted from the config slug.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum Family {
    Cosine,
    ModifiedCosine,
    Entropy,
    ModifiedEntropy,
}

impl Family {
    pub const ALL: [Self; 4] = [
        Self::Cosine,
        Self::ModifiedCosine,
        Self::Entropy,
        Self::ModifiedEntropy,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Cosine => "Cosine",
            Self::ModifiedCosine => "Modified cosine",
            Self::Entropy => "Entropy",
            Self::ModifiedEntropy => "Modified entropy",
        }
    }
}

/// Compact floating-point key used in the slug, kept as a fixed-point
/// integer (thousandths) so we can put `Family + ExpKey + ExpKey + Option<bool>`
/// in a `HashMap` / `BTreeSet`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct ExpKey(i32);

impl ExpKey {
    pub fn from_f64(value: f64) -> Self {
        Self((value * 1000.0).round() as i32)
    }

    pub fn as_f64(self) -> f64 {
        f64::from(self.0) / 1000.0
    }

    pub fn label(self) -> String {
        let v = self.as_f64();
        if (v - v.round()).abs() < 1.0e-6 {
            format!("{v:.0}")
        } else {
            format!("{v}")
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct ConfigKey {
    pub family: Family,
    pub mz_exp: ExpKey,
    pub int_exp: ExpKey,
    pub weighted: Option<bool>,
}

impl ConfigKey {
    /// Parse a slug like `cosine_mz0.000_int1.000` or
    /// `entropy_mz0.000_int1.000_weightedfalse` into a structured key.
    /// Returns `None` if the prefix is unknown.
    pub fn parse(slug: &str) -> Option<Self> {
        let (family, rest) = if let Some(rest) = slug.strip_prefix("modified_entropy_") {
            (Family::ModifiedEntropy, rest)
        } else if let Some(rest) = slug.strip_prefix("entropy_") {
            (Family::Entropy, rest)
        } else if let Some(rest) = slug.strip_prefix("modified_cosine_") {
            (Family::ModifiedCosine, rest)
        } else {
            let rest = slug.strip_prefix("cosine_")?;
            (Family::Cosine, rest)
        };
        let mut mz: Option<f64> = None;
        let mut int_v: Option<f64> = None;
        let mut weighted: Option<bool> = None;
        for part in rest.split('_') {
            if let Some(value) = part.strip_prefix("mz") {
                mz = value.parse().ok();
            } else if let Some(value) = part.strip_prefix("int") {
                int_v = value.parse().ok();
            } else if let Some(value) = part.strip_prefix("weighted") {
                weighted = Some(value == "true");
            }
        }
        Some(Self {
            family,
            mz_exp: ExpKey::from_f64(mz?),
            int_exp: ExpKey::from_f64(int_v?),
            weighted,
        })
    }

    /// Inverse of [`Self::parse`]: rebuild the original config slug.
    pub fn slug(self) -> String {
        let family_prefix = match self.family {
            Family::Cosine => "cosine",
            Family::ModifiedCosine => "modified_cosine",
            Family::Entropy => "entropy",
            Family::ModifiedEntropy => "modified_entropy",
        };
        let weighted_suffix = match self.weighted {
            Some(true) => "_weightedtrue",
            Some(false) => "_weightedfalse",
            None => "",
        };
        format!(
            "{family_prefix}_mz{:.3}_int{:.3}{weighted_suffix}",
            self.mz_exp.as_f64(),
            self.int_exp.as_f64(),
        )
    }
}

/// Lookup of every available config slug → (`config_index`, key).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigCatalog {
    by_index: Vec<(String, ConfigKey)>,
}

impl ConfigCatalog {
    pub fn new<'a>(configs: impl IntoIterator<Item = &'a (usize, String)>) -> Self {
        let mut by_index: Vec<(String, ConfigKey)> = Vec::new();
        let mut as_vec: Vec<(usize, String)> = configs.into_iter().cloned().collect();
        as_vec.sort_by_key(|(i, _)| *i);
        by_index.reserve(as_vec.len());
        for (_, slug) in as_vec {
            if let Some(key) = ConfigKey::parse(&slug) {
                by_index.push((slug, key));
            }
        }
        Self { by_index }
    }

    /// Resolve a structured key to the `config_index` axis used by the npz.
    pub fn index_for(&self, key: &ConfigKey) -> Option<usize> {
        self.by_index.iter().position(|(_, k)| k == key)
    }

    pub fn slug(&self, index: usize) -> Option<&str> {
        self.by_index.get(index).map(|(s, _)| s.as_str())
    }

    pub fn families(&self) -> BTreeSet<Family> {
        self.by_index.iter().map(|(_, k)| k.family).collect()
    }

    pub fn mz_for(&self, family: Family) -> BTreeSet<ExpKey> {
        self.by_index
            .iter()
            .filter(|(_, k)| k.family == family)
            .map(|(_, k)| k.mz_exp)
            .collect()
    }

    pub fn int_for(&self, family: Family, mz_exp: ExpKey) -> BTreeSet<ExpKey> {
        self.by_index
            .iter()
            .filter(|(_, k)| k.family == family && k.mz_exp == mz_exp)
            .map(|(_, k)| k.int_exp)
            .collect()
    }

    /// Whether the family supports a `weighted` flag at all.
    pub fn supports_weighted(&self, family: Family) -> bool {
        self.by_index
            .iter()
            .any(|(_, k)| k.family == family && k.weighted.is_some())
    }

    /// Find any valid config that matches `family`, falling back if the
    /// requested mz/int/weighted combination doesn't exist.
    pub fn closest(&self, family: Family) -> Option<ConfigKey> {
        self.by_index
            .iter()
            .find(|(_, k)| k.family == family)
            .map(|(_, k)| *k)
    }

    pub fn closest_for_mz(&self, family: Family, mz_exp: ExpKey) -> Option<ConfigKey> {
        self.by_index
            .iter()
            .find(|(_, k)| k.family == family && k.mz_exp == mz_exp)
            .map(|(_, k)| *k)
    }

    pub fn first(&self) -> Option<ConfigKey> {
        self.by_index.first().map(|(_, k)| *k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_cosine() {
        let k = ConfigKey::parse("cosine_mz0.000_int1.000").unwrap();
        assert_eq!(k.family, Family::Cosine);
        assert!((k.mz_exp.as_f64() - 0.0).abs() < 1.0e-9);
        assert!((k.int_exp.as_f64() - 1.0).abs() < 1.0e-9);
        assert_eq!(k.weighted, None);
    }

    #[test]
    fn parse_entropy() {
        let k = ConfigKey::parse("modified_entropy_mz0.000_int1.000_weightedfalse").unwrap();
        assert_eq!(k.family, Family::ModifiedEntropy);
        assert_eq!(k.weighted, Some(false));
    }
}
