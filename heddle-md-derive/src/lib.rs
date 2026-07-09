//! Derive macro for `crate::units::Convert`.
//!
//! Generates `impl Convert for T` such that each field of `T` (across
//! all struct fields or enum variants) has `Convert::from_user(&mut
//! field, units)` and `Convert::to_user(&mut field, units)` called on
//! it. Recurses through the type via the existing `Convert` impls on
//! the dimensioned newtypes (`Length`, `Energy`, …) and the no-op
//! impls on `String`, `bool`, integers, `f64`, etc. (see
//! `src/units/mod.rs`).
//!
//! Has no helper attributes: every field of every variant must
//! implement `Convert`. Empty unit variants and unit structs are
//! handled (they contain no fields and so their match arm does
//! nothing).

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields, parse_macro_input};

#[proc_macro_derive(Convert)]
pub fn convert_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident.clone();
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let (from_body, to_body) = match &input.data {
        Data::Struct(s) => struct_bodies(&s.fields),
        Data::Enum(e) => enum_bodies(e),
        Data::Union(_) => panic!("`#[derive(Convert)]` does not support union types"),
    };

    let expanded = quote! {
        impl #impl_generics crate::units::Convert for #name #ty_generics #where_clause {
            fn from_user(&mut self, units: crate::units::UnitSystem) {
                #from_body
            }
            fn to_user(&mut self, units: crate::units::UnitSystem) {
                #to_body
            }
        }
    };
    expanded.into()
}

fn struct_bodies(fields: &Fields) -> (TokenStream2, TokenStream2) {
    match fields {
        Fields::Named(named) => {
            let names: Vec<_> = named.named.iter().map(|f| f.ident.clone().unwrap()).collect();
            let from = quote! {
                #( crate::units::Convert::from_user(&mut self.#names, units); )*
            };
            let to = quote! {
                #( crate::units::Convert::to_user(&mut self.#names, units); )*
            };
            (from, to)
        }
        Fields::Unnamed(unnamed) => {
            let idxs: Vec<_> = (0..unnamed.unnamed.len())
                .map(syn::Index::from)
                .collect();
            let from = quote! {
                #( crate::units::Convert::from_user(&mut self.#idxs, units); )*
            };
            let to = quote! {
                #( crate::units::Convert::to_user(&mut self.#idxs, units); )*
            };
            (from, to)
        }
        Fields::Unit => (quote! {}, quote! {}),
    }
}

fn enum_bodies(e: &syn::DataEnum) -> (TokenStream2, TokenStream2) {
    let mut from_arms = Vec::new();
    let mut to_arms = Vec::new();
    for variant in &e.variants {
        let v_ident = &variant.ident;
        match &variant.fields {
            Fields::Named(named) => {
                let names: Vec<_> = named
                    .named
                    .iter()
                    .map(|f| f.ident.clone().unwrap())
                    .collect();
                from_arms.push(quote! {
                    Self::#v_ident { #(#names),* } => {
                        #( crate::units::Convert::from_user(#names, units); )*
                    }
                });
                to_arms.push(quote! {
                    Self::#v_ident { #(#names),* } => {
                        #( crate::units::Convert::to_user(#names, units); )*
                    }
                });
            }
            Fields::Unnamed(unnamed) => {
                let bindings: Vec<_> = (0..unnamed.unnamed.len())
                    .map(|i| format_ident!("__f{}", i))
                    .collect();
                from_arms.push(quote! {
                    Self::#v_ident( #(#bindings),* ) => {
                        #( crate::units::Convert::from_user(#bindings, units); )*
                    }
                });
                to_arms.push(quote! {
                    Self::#v_ident( #(#bindings),* ) => {
                        #( crate::units::Convert::to_user(#bindings, units); )*
                    }
                });
            }
            Fields::Unit => {
                from_arms.push(quote! { Self::#v_ident => {} });
                to_arms.push(quote! { Self::#v_ident => {} });
            }
        }
    }
    let from = quote! { match self { #(#from_arms),* } };
    let to = quote! { match self { #(#to_arms),* } };
    (from, to)
}
