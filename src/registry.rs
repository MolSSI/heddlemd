// rq-54fc86d6
//
// Generic host-side builder registry shared by every open-extensible
// builder collection in the engine (integrators, thermostats,
// barostats, constraint types, potentials, minimizers, analyses). See
// `rqm/registry-framework.md`.

use std::fmt;

/// Keyed-lookup capability carried by every named-selection builder
/// trait (the six that select a single builder by a TOML `kind`
/// string). `PotentialBuilder` deliberately does not carry it, which is
/// what withholds `Registry::lookup` from `PotentialRegistry`.
// rq-0f6b7b7a
pub trait KindedBuilder {
    fn kind_name(&self) -> &'static str;

    /// Rescale this kind's open-shaped `params` table from the user's
    /// unit system to atomic units, in place. The default is a no-op
    /// (appropriate for a kind with no unit-bearing params); builders
    /// with unit-bearing params override it, conventionally via
    /// [`convert_params_in_place`] applied to their typed parameter
    /// struct (which derives `Convert`). See `rqm/io/unit-system.md`.
    fn convert_params(
        &self,
        _units: crate::units::UnitSystem,
        _params: &mut toml::Value,
    ) -> Result<(), crate::io::config::ConfigError> {
        Ok(())
    }
}

/// Deserialise `params` into the typed parameter struct `P` (which
/// derives `Convert`), rescale it from the user's unit system to atomic
/// units, and write the converted values back into `params`. Existing
/// keys are overwritten by their converted values; any key not modelled
/// by `P` is preserved. A `params` table that does not deserialise into
/// `P` is left untouched and returns `Ok(())`, deferring the typed error
/// to the builder's `validate_params`. rq-0f6b7b7a
pub fn convert_params_in_place<P>(
    units: crate::units::UnitSystem,
    params: &mut toml::Value,
) -> Result<(), crate::io::config::ConfigError>
where
    P: serde::de::DeserializeOwned + serde::Serialize + crate::units::Convert,
{
    let mut typed: P = match params.clone().try_into() {
        Ok(p) => p,
        Err(_) => return Ok(()),
    };
    typed.from_user(units);
    let converted = toml::Value::try_from(&typed).map_err(|e| {
        crate::io::config::ConfigError::Parse {
            path: "constraint/slot params".to_string(),
            message: e.to_string(),
        }
    })?;
    // Overwrite only the keys the user actually wrote: a field omitted
    // from the input (filled by a serde default) must stay absent so the
    // builder's own default applies, matching the field-table behaviour.
    if let (Some(dst), Some(src)) = (params.as_table_mut(), converted.as_table()) {
        for (k, v) in src {
            if dst.contains_key(k) {
                dst.insert(k.clone(), v.clone());
            }
        }
    }
    Ok(())
}

/// Per-trait built-in roster. Implemented once per builder trait object
/// (e.g. `impl Builtins for dyn IntegratorBuilder`); `Registry::with_builtins`
/// and the `Default` impl read it.
// rq-c00689e6
pub trait Builtins {
    fn builtins() -> Vec<Box<Self>>;
}

/// Generates the boxed-clone plumbing that makes a `Registry<dyn $bt>`
/// `Clone` without a per-builder clone method: a helper supertrait
/// carrying `registry_clone_box`, a blanket impl over `Clone` builders,
/// and `impl Clone for Box<dyn $bt>`. A concrete builder needs only
/// `#[derive(Clone)]`.
///
/// The helper is a non-generic supertrait (its trait-object type appears
/// only in the method return, never as a trait type parameter), which
/// avoids the super-predicate cycle a `CloneToBox<dyn $bt>` supertrait
/// would create. See `rqm/registry-framework.md`.
// rq-b775df32
#[macro_export]
macro_rules! registry_builder_clone {
    ($vis:vis $helper:ident for $bt:ident) => {
        $vis trait $helper {
            fn registry_clone_box(&self) -> ::std::boxed::Box<dyn $bt>;
        }
        impl<T: $bt + ::core::clone::Clone + 'static> $helper for T {
            fn registry_clone_box(&self) -> ::std::boxed::Box<dyn $bt> {
                ::std::boxed::Box::new(::core::clone::Clone::clone(self))
            }
        }
        impl ::core::clone::Clone for ::std::boxed::Box<dyn $bt> {
            fn clone(&self) -> Self {
                self.registry_clone_box()
            }
        }
    };
}

/// Generic host-side container of boxed builders. Concrete registries
/// are type aliases for this specialised to one builder trait object
/// (e.g. `pub type IntegratorRegistry = Registry<dyn IntegratorBuilder>`).
// rq-e0ea3802
pub struct Registry<B: ?Sized> {
    builders: Vec<Box<B>>,
}

impl<B: ?Sized + fmt::Debug> fmt::Debug for Registry<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Registry")
            .field("builders", &self.builders)
            .finish()
    }
}

impl<B: ?Sized> Registry<B> {
    /// An empty registry (no builders).
    pub fn new() -> Self {
        Registry { builders: Vec::new() }
    }

    /// A registry holding the given builders in the given order.
    pub fn from_builders(builders: Vec<Box<B>>) -> Self {
        Registry { builders }
    }

    /// Appends `builder`, preserving registration order.
    pub fn register(&mut self, builder: Box<B>) {
        self.builders.push(builder);
    }

    /// The held builders in registration order, for subsystems that
    /// iterate every builder (compositional activation).
    pub fn builders(&self) -> &[Box<B>] {
        &self.builders
    }
}

impl<B: ?Sized + KindedBuilder> Registry<B> {
    /// The first held builder whose `kind_name()` equals `kind` in
    /// registration order, or `None` if no builder matches. Available
    /// only for named-selection registries.
    pub fn lookup(&self, kind: &str) -> Option<&B> {
        self.builders
            .iter()
            .map(|b| b.as_ref())
            .find(|b| b.kind_name() == kind)
    }
}

impl<B: ?Sized + Builtins> Registry<B> {
    /// A registry pre-populated with `B::builtins()` in canonical order.
    pub fn with_builtins() -> Self {
        Registry { builders: B::builtins() }
    }
}

impl<B: ?Sized + Builtins> Default for Registry<B> {
    fn default() -> Self {
        Registry::with_builtins()
    }
}

impl<B: ?Sized> Clone for Registry<B>
where
    Box<B>: Clone,
{
    fn clone(&self) -> Self {
        Registry {
            builders: self.builders.clone(),
        }
    }
}
