/// Install the panic hook once. Called from `LunaVDB::new` so that if we ever
/// do hit an internal invariant, the browser/devtools console shows the Rust
/// message instead of a bare `unreachable executed`.
pub fn set_panic_hook() {
    #[cfg(all(feature = "console_error_panic_hook", target_arch = "wasm32"))]
    {
        use std::sync::Once;
        static ONCE: Once = Once::new();
        ONCE.call_once(console_error_panic_hook::set_once);
    }
}
